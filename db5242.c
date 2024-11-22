/*
  CSE 5242 Project 2, Fall 2024

  See class project handout for more extensive documentation.
*/

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <unistd.h>
#include <immintrin.h>

/* uncomment out the following DEBUG line for debug info, for experiment comment the DEBUG line  */
// #define DEBUG

/* compare two int64_t values - for use with qsort */
static int compare(const void *p1, const void *p2)
{
  int a,b;
  a = *(int64_t *)p1;
  b = *(int64_t *)p2;
  if (a<b) return -1;
  if (a==b) return 0;
  return 1;
}

/* initialize searches and data - data is sorted and searches is a random permutation of data */
int init(int64_t* data, int64_t* searches, int count)
{
  for(int64_t i=0; i<count; i++){
    searches[i] = random();
    data[i] = searches[i]+1;
  }
  qsort(data,count,sizeof(int64_t),compare);
}

/* initialize outer probes of band join */
int band_init(int64_t* outer, int64_t size)
{
  for(int64_t i=0; i<size; i++){
    outer[i] = random();
  }
}

inline int64_t simple_binary_search(int64_t* data, int64_t size, int64_t target)
{
  int64_t left=0;
  int64_t right=size;
  int64_t mid;

  while(left<=right) {
    mid = (left + right)/2;   /* ignore possibility of overflow of left+right */
    if (data[mid]==target) return mid;
    if (data[mid]<target) left=mid+1;
    else right = mid-1;
  }
  return -1; /* no match */
}

inline int64_t low_bin_search(int64_t* data, int64_t size, int64_t target)
{
  /* this binary search variant
     (a) does only one comparison in the inner loop
     (b) doesn't require an exact match; instead it returns the index of the first key >= the search key.
         That's good in a DB context where we might be doing a range search, and using binary search to
	 identify the first key in the range.
     (c) If the search key is bigger than all keys, it returns size.
  */
  int64_t left=0;
  int64_t right=size;
  int64_t mid;

  while(left<right) {
    mid = (left + right)/2;   /* ignore possibility of overflow of left+right */
    if (data[mid]>=target)
      right=mid;
    else
      left=mid+1;
  }
  return right;
}

inline int64_t low_bin_nb_arithmetic(int64_t* data, int64_t size, int64_t target)
{
  /* this binary search variant
     (a) does no comparisons in the inner loop by using multiplication and addition to convert control dependencies
         to data dependencies
     (b) doesn't require an exact match; instead it returns the index of the first key >= the search key.
         That's good in a DB context where we might be doing a range search, and using binary search to
	 identify the first key in the range.
     (c) If the search key is bigger than all keys, it returns size.
  */
  int64_t left=0;
  int64_t right=size;
  int64_t mid;


  while(left<right) {
    mid = (left + right) >> 1;
    int64_t GE = (data[mid] >= target);
    right = (GE * mid) + (!GE * right);
    left = (!GE * (mid+1)) + (GE * left);
  }
  return right;
}

inline int64_t low_bin_nb_mask(int64_t* data, int64_t size, int64_t target)
{
  /* this binary search variant
     (a) does no comparisons in the inner loop by using bit masking operations to convert control dependencies
         to data dependencies
     (b) doesn't require an exact match; instead it returns the index of the first key >= the search key.
         That's good in a DB context where we might be doing a range search, and using binary search to
	 identify the first key in the range.
     (c) If the search key is bigger than all keys, it returns size.
  */
  int64_t left=0;
  int64_t right=size;
  int64_t mid;

   while(left<right){
    mid = (left + right) >> 1;
    int64_t GE = -(data[mid] >= target);
    right = (mid & GE) | (right & ~GE);
    left = ((mid + 1) & ~GE) | (left & GE);
  }


  return right;
}

inline void low_bin_nb_4x(int64_t* data, int64_t size, int64_t* targets, int64_t* right)
{
  /* this binary search variant
     (a) does no comparisons in the inner loop by using bit masking operations instead
     (b) doesn't require an exact match; instead it returns the index of the first key >= the search key.
         That's good in a DB context where we might be doing a range search, and using binary search to
	 identify the first key in the range.
     (c) If the search key is bigger than all keys, it returns size.
     (d) does 4 searches at the same time in an interleaved fashion, so that an out-of-order processor
         can make progress even when some instructions are still waiting for their operands to be ready.

     Note that we're using the result array as the "right" elements in the search so no need for a return statement
  */

  int64_t left[4] = {0,0,0,0};
  int64_t mid[4];
  for(int i=0; i<4; i++) {
    right[i] = size;
  }

  while(left[0]<right[0] || left[1]<right[1] || left[2]<right[2] || left[3]<right[3]){
    for(int i=0; i<4; i++){
      mid[i] = (left[i] + right[i]) >> 1;
      int64_t GE = -((data[mid[i]] >= targets[i]));
      right[i] = (mid[i] & GE) | (right[i] & ~GE);
      left[i] = ((mid[i] + 1) & ~GE) | (left[i] & GE);
    }
  }

}


/* The following union type is handy to output the contents of AVX512 data types */
union int8x4 {
  __m256i a;
  int64_t b[4];
};

void printavx(char* name, __m256i v) {
  union int8x4 n;

  n.a=v;
  printf("Value in %s is [%ld %ld %ld %ld ]\n",name,n.b[0],n.b[1],n.b[2],n.b[3]);
}

/*
 * Optinal for using AVX-512

  union int8x8 {
    __m512i a;
    int64_t b[8];
  };

  void printavx512(char* name, __m512i v) {
    union int8x4 n;

    n.a=v;
    printf("Value in %s is [%ld %ld %ld %ld %ld %ld %ld %ld ]\n",name,n.b[0],n.b[1],n.b[2],n.b[3]);
  }

 */


inline void low_bin_nb_simd(int64_t* data, int64_t size, __m256i target, __m256i* result)
{
  /* this binary search variant
     (a) does no comparisons in the inner loop by using bit masking operations instead
     (b) doesn't require an exact match; instead it returns the index of the first key >= the search key.
         That's good in a DB context where we might be doing a range search, and using binary search to
	 identify the first key in the range.
     (c) If the search key is bigger than all keys, it returns size.
     (d) does 4 searches at the same time using AVX2 intrinsics

     See https://www.intel.com/content/www/us/en/docs/cpp-compiler/developer-guide-reference/2021-8/intrinsics-for-avx2.html
     for documentation of the AVX512 intrinsics

     Note that we're using the result array as the "right" elements in the search, and that searchkey is being passed
     as an __m256i value rather than via a pointer.
  */


   __m256i left = _mm256_setzero_si256();
  __m256i right = _mm256_set1_epi64x(size);
  __m256i mid, GE, LT;
  while(_mm256_movemask_epi8(_mm256_cmpgt_epi64(right,left))!= 0){ // compare each 64-bit element in left and right until all left >= right
    // printf("in while...\n");
    mid = _mm256_srli_epi64(_mm256_add_epi64(left,right),1); // mid = (left + right) >> 1;
    __m256i midVal = _mm256_i64gather_epi64((const long long int *)data,mid,8); // midVal = data[mid];
    __m256i greater = _mm256_cmpgt_epi64(midVal,target); // GE = (midVal > target);
    __m256i equal = _mm256_cmpeq_epi64(midVal,target); // EQ = (midVal == target);
    GE = _mm256_or_si256(greater,equal); // GE = GE | EQ;
    LT = _mm256_xor_si256(GE,_mm256_set1_epi64x(-1)); // LT = ~GE;
    right = _mm256_or_si256(_mm256_and_si256(mid,GE),_mm256_and_si256(right,LT)); // right = (mid & GE) | (right & LT);
    left = _mm256_or_si256(_mm256_and_si256(_mm256_add_epi64(mid,_mm256_set1_epi64x(1)),LT),_mm256_and_si256(left,GE)); // left = ((mid + 1) & LT) | (left & GE);
  }

  *result = right;
}

void bulk_bin_search(int64_t* data, int64_t size, int64_t* searchkeys, int64_t numsearches, int64_t* results, int repeats)
{
  for(int j=0; j<repeats; j++) {
    /* Function to test a large number of binary searches

       we might need repeats>1 to make sure the events we're measuring are not dominated by various
       overheads, particularly for small values of size and/or numsearches

       we assume that we want exactly "size" searches, where "size" is the length if the searchkeys array
     */
    for(int64_t i=0;i<numsearches; i++) {
#ifdef DEBUG
      printf("Searching for %ld...\n",searchkeys[i]);
#endif

      // Uncomment one of the following to measure it
      //results[i] = low_bin_search(data,size,searchkeys[i]);
      //results[i] = low_bin_nb_arithmetic(data,size,searchkeys[i]);
      results[i] = low_bin_nb_mask(data,size,searchkeys[i]);

#ifdef DEBUG
      printf("Result is %ld\n",results[i]);
      printf("Value is %ld\n",data[results[i]]);
#endif
    }
  }
}

void bulk_bin_search_4x(int64_t* data, int64_t size, int64_t* searchkeys, int64_t numsearches, int64_t* results, int repeats)
{
  register __m256i searchkey_4x;

  for(int j=0; j<repeats; j++) {
    /* Function to test a large number of binary searches using one of the 8x routines

       we might need repeats>1 to make sure the events we're measuring are not dominated by various
       overheads, particularly for small values of size and/or numsearches

       we assume that we want exactly "size" searches, where "size" is the length if the searchkeys array
     */
    int64_t extras = numsearches % 4;
    for(int64_t i=0;i<numsearches-extras; i+=4) {
#ifdef DEBUG
      printf("Searching for %ld %ld %ld %ld  ...\n",
	     searchkeys[i],searchkeys[i+1],searchkeys[i+2],searchkeys[i+3]);
#endif

      // Uncomment one of the following depending on which routine you want to profile

      // Algorithm A
      // low_bin_nb_4x(data,size,&searchkeys[i],&results[i]);

      // Algorithm B
      searchkey_4x = _mm256_loadu_si256((__m256i *)&searchkeys[i]);
      low_bin_nb_simd(data,size,searchkey_4x,(__m256i *)&results[i]);

#ifdef DEBUG
      printf("Result is %ld %ld %ld %ld  ...\n",
	     results[i],results[i+1],results[i+2],results[i+3]);
      printf("Values are %ld %ld %ld %ld  ...\n",
	     data[results[i]],data[results[i+1]],data[results[i+2]],data[results[i+3]]);
#endif
    }
    /* a little bit more work if numsearches is not a multiple of 8 */
    for(int64_t i=numsearches-extras;i<numsearches; i++) {

      results[i] = low_bin_nb_mask(data,size,searchkeys[i]);

    }

  }
}


int64_t band_join(int64_t* inner, int64_t inner_size, int64_t* outer, int64_t outer_size, int64_t* inner_results, int64_t* outer_results, int64_t result_size, int64_t bound)
{
  /* In a band join we want matches within a range of values.  If p is the probe value from the outer table, then all
     reccords in the inner table with a key in the range [p-bound,p+bound] inclusive should be part of the result.

     Results are returned via two arrays. outer_results stores the index of the outer table row that matches, and
     inner_results stores the index of the inner table row that matches.  result_size tells you the size of the
     output array that has been allocated. You should make sure that you don't exceed this size.  If there are
     more results than can fit in the result arrays, then return early with just a prefix of the results in the result
     arrays. The return value of the function should be the number of output results.

  */

  /* YOUR CODE HERE */

  int64_t current_index = 0;
  int64_t extra_searches = outer_size % 4;  // Search all values of outer array using low_bin_nb_4x until less than 4 searches remain

  for (int64_t i = 0; i < (outer_size - extra_searches); i+=4) {
    // use 4x search here
    int64_t results[4] = {inner_size, inner_size, inner_size, inner_size};
    low_bin_nb_4x(inner, inner_size, &outer[i], &results[0]);
    // returns indices in results s.t. outer[i] <= inner[results[0]], outer[i+1] <= inner[results[1]], ...
    
    for (int j = 0; j < 4; j++) {
      int64_t left = results[j] - 1;
      int64_t right = results[j];
      // Should we expand the whiles to do all 4 elements at a time? Not sure it would be faster...
      while ( (outer[i+j] - bound < inner[left]) && (left >= 0) && (current_index < result_size)) {
        // first go left until outer[i] - bound > inner[left]
        inner_results[current_index] = left--;
        outer_results[current_index] = i+j;
        current_index++;
      }

      while ( (outer[i+j] + bound > inner[right]) && (right < inner_size) && (current_index < result_size)) {
        // now go right until outer[i] + bound < inner[right]
        inner_results[current_index] = right++;
        outer_results[current_index] = i+j;
        current_index++;
        
      }

    }
    
    if (current_index >= result_size) break;
  }

  for (int64_t i = outer_size - extra_searches; i < outer_size; i++) {
    // printf("Entering cleanup, current_index=%ld\n", current_index);
    if (current_index >= result_size) break;
    // use mask search here
    int64_t right = low_bin_nb_mask(inner, inner_size, outer[i]);
    int64_t left = right - 1;
    while ( (outer[i] - bound < inner[left]) && (left >= 0) && (current_index < result_size)) {
      // first go left until outer[i] - bound > inner[left]
      inner_results[current_index] = left--;
      outer_results[current_index++] = i;
    }
    while ( (outer[i] + bound > inner[right]) && (right < inner_size) && (current_index < result_size)) {
      // now go right until outer[i] + bound < inner[right]
      inner_results[current_index] = right++;
      outer_results[current_index++] = i;
    }
  }
  return current_index;
}

int64_t band_join_simd(int64_t* inner, int64_t inner_size, int64_t* outer, int64_t outer_size, int64_t* inner_results, int64_t* outer_results, int64_t result_size, int64_t bound)
{
  /* In a band join we want matches within a range of values.  If p is the probe value from the outer table, then all
     reccords in the inner table with a key in the range [p-bound,p+bound] inclusive should be part of the result.

     Results are returned via two arrays. outer_results stores the index of the outer table row that matches, and
     inner_results stores the index of the inner table row that matches.  result_size tells you the size of the
     output array that has been allocated. You should make sure that you don't exceed this size.  If there are
     more results than can fit in the result arrays, then return early with just a prefix of the results in the result
     arrays. The return value of the function should be the number of output results.

     To do the binary search, you could use the low_bin_nb_simd you just implemented to search for the lower bounds in parallel

     Once you've found the lower bounds, do the following for each of the 4 search keys in turn:
        scan along the sorted inner array, generating outputs for each match, and making sure not to exceed the output array bounds.

     This inner scanning code does not have to use SIMD.
  */

      /* YOUR CODE HERE */

  int64_t current_index = 0;
  int64_t extra_searches = outer_size % 4;  // Search all values of outer array using low_bin_nb_4x until less than 4 searches remain
  register __m256i searchkeys;

  for (int64_t i = 0; i < (outer_size - extra_searches); i+=4) {
    // use SIMD search here
    int64_t results[4] = {5,5,5,5};// {inner_size, inner_size, inner_size, inner_size};

    // Load 4 outer keys into vector for SIMD search.
    searchkeys = _mm256_loadu_si256((__m256i *)&outer[i]);
    low_bin_nb_simd(inner, inner_size, searchkeys, (__m256i *)&results[0]);
    // returns indices in results s.t. outer[i] <= inner[results[0]], outer[i+1] <= inner[results[1]], ...

    #ifdef DEBUG
      printf("Result is %ld %ld %ld %ld  ...\n",
        results[0],results[1],results[2],results[3]);
      printf("Values are %ld %ld %ld %ld  ...\n",
        inner[results[0]],inner[results[1]],inner[results[2]],inner[results[3]]);
    #endif

    // Iterate through the returned values.  Complexity = O(S1+S2+S3+S4)
    for (int j = 0; j < 4; j++) {
      int64_t left = results[j] - 1;
      int64_t right = results[j];
      // Scan through the adjacent values to find all entries that fall within the bound.
      while ( (outer[i+j] - bound < inner[left]) && (left >= 0) && (current_index < result_size)) {
        // first go left until outer[i] - bound > inner[left]
        inner_results[current_index] = left--;
        outer_results[current_index] = i+j;
        current_index++;
      }

      while ( (outer[i+j] + bound > inner[right]) && (right < inner_size) && (current_index < result_size)) {
        // now go right until outer[i] + bound < inner[right]
        inner_results[current_index] = right++;
        outer_results[current_index] = i+j;
        current_index++;
        
      }

    }
    // If we've hit the max result size, quit.
    if (current_index >= result_size) break;
  }

  // Do some extra searches if not perfectly divisible by 4.
  for (int64_t i = outer_size - extra_searches; i < outer_size; i++) {
    if (current_index >= result_size) break;
    // use mask search here
    int64_t right = low_bin_nb_mask(inner, inner_size, outer[i]);
    int64_t left = right - 1;
    
    // Scan through the adjacent values to find all entries that fall within the bound.
    while ( (outer[i] - bound < inner[left]) && (left >= 0) && (current_index < result_size)) {
      // first go left until outer[i] - bound > inner[left]
      inner_results[current_index] = left--;
      outer_results[current_index++] = i;
    }
    while ( (outer[i] + bound > inner[right]) && (right < inner_size) && (current_index < result_size)) {
      // now go right until outer[i] + bound < inner[right]
      inner_results[current_index] = right++;
      outer_results[current_index++] = i;
    }
  }
  return current_index;

}

int
main(int argc, char *argv[])
{
	 long long counter;
	 int64_t arraysize, outer_size, result_size;
	 int64_t bound;
	 int64_t *data, *queries, *results;
	 int ret;
	 struct timeval before, after;
	 int repeats;
	 int64_t total_results;

	 // band-join arrays
	 int64_t *outer, *outer_results, *inner_results;


	 if (argc >= 5)
	   {
	     arraysize = atoi(argv[1]);
	     outer_size = atoi(argv[2]);
	     result_size = atoi(argv[3]);
	     bound = atoi(argv[4]);
	   }
	 else
	   {
	     fprintf(stderr, "Usage: db5242 inner_size outer_size result_size bound <repeats>\n");
	     exit(EXIT_FAILURE);
	   }

	 if (argc >= 6)
	   repeats = atoi(argv[5]);
	 else
	   {
	     repeats=1;
	   }

	 /* allocate the array and the queries for searching */
	 ret=posix_memalign((void**) &data,64,arraysize*sizeof(int64_t));
	 if (ret)
	 {
	   fprintf(stderr, "Memory allocation error.\n");
	   exit(EXIT_FAILURE);
	 }
	 ret=posix_memalign((void**) &queries,64,arraysize*sizeof(int64_t));
	 if (ret)
	 {
	   fprintf(stderr, "Memory allocation error.\n");
	   exit(EXIT_FAILURE);
	 }
	 ret=posix_memalign((void**) &results,64,arraysize*sizeof(int64_t));
	 if (ret)
	 {
	   fprintf(stderr, "Memory allocation error.\n");
	   exit(EXIT_FAILURE);
	 }

	 /* allocate the outer array and output arrays for band-join */
	 ret=posix_memalign((void**) &outer,64,outer_size*sizeof(int64_t));
	 if (ret)
	 {
	   fprintf(stderr, "Memory allocation error.\n");
	   exit(EXIT_FAILURE);
	 }
	 ret=posix_memalign((void**) &outer_results,64,result_size*sizeof(int64_t));
	 if (ret)
	 {
	   fprintf(stderr, "Memory allocation error.\n");
	   exit(EXIT_FAILURE);
	 }
	 ret=posix_memalign((void**) &inner_results,64,result_size*sizeof(int64_t));
	 if (ret)
	 {
	   fprintf(stderr, "Memory allocation error.\n");
	   exit(EXIT_FAILURE);
	 }


	   /* code to initialize data structures goes here so that it is not included in the timing measurement */
	   init(data,queries,arraysize);
	   band_init(outer,outer_size);

#ifdef DEBUG
	   /* show the arrays */
	   printf("data: ");
	   for(int64_t i=0;i<arraysize;i++) printf("%ld ",data[i]);
	   printf("\n");
	   printf("queries: ");
	   for(int64_t i=0;i<arraysize;i++) printf("%ld ",queries[i]);
	   printf("\n");
	   printf("outer: ");
	   for(int64_t i=0;i<outer_size;i++) printf("%ld ",outer[i]);
	   printf("\n");
#endif


	   /* now measure... */

	   gettimeofday(&before,NULL);

	   /* the code that you want to measure goes here; make a function call */
	   //bulk_bin_search(data,arraysize,queries,arraysize,results, repeats);

	   gettimeofday(&after,NULL);
	   printf("Time in bulk_bin_search loop is %ld microseconds or %f microseconds per search\n", (after.tv_sec-before.tv_sec)*1000000+(after.tv_usec-before.tv_usec), 1.0*((after.tv_sec-before.tv_sec)*1000000+(after.tv_usec-before.tv_usec))/arraysize/repeats);



	   gettimeofday(&before,NULL);

	   /* the code that you want to measure goes here; make a function call */
	   bulk_bin_search_4x(data,arraysize,queries,arraysize,results, repeats);

	   gettimeofday(&after,NULL);
	   printf("Time in bulk_bin_search_4x loop is %ld microseconds or %f microseconds per search\n", (after.tv_sec-before.tv_sec)*1000000+(after.tv_usec-before.tv_usec), 1.0*((after.tv_sec-before.tv_sec)*1000000+(after.tv_usec-before.tv_usec))/arraysize/repeats);


	   gettimeofday(&before,NULL);

	   /* the code that you want to measure goes here; make a function call */
	   //total_results=band_join(data, arraysize, outer, outer_size, inner_results, outer_results, result_size, bound);
     total_results=band_join_simd(data, arraysize, outer, outer_size, inner_results, outer_results, result_size, bound);

	   gettimeofday(&after,NULL);
	   printf("Band join result size is %ld with an average of %f matches per output record\n",total_results, 1.0*total_results/(1.0+outer_results[total_results-1]));
	   printf("Time in band_join loop is %ld microseconds or %f microseconds per outer record\n", (after.tv_sec-before.tv_sec)*1000000+(after.tv_usec-before.tv_usec), 1.0*((after.tv_sec-before.tv_sec)*1000000+(after.tv_usec-before.tv_usec))/outer_size);

#ifdef DEBUG
	   /* show the band_join results */
	   printf("band_join results: ");
	   for(int64_t i=0;i<total_results;i++) printf("(%ld,%ld) ",outer_results[i],inner_results[i]);
     printf("\nresult values:\n");
     for(int64_t i=0;i<total_results;i++) printf("(%ld,%ld) ",outer[outer_results[i]],data[inner_results[i]]);
	   printf("\n");
#endif


}