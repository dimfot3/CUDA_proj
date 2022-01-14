/**
 * Author: Fotiou Dimitrios
 * AEM: 9650
 * Here there are tests to evaluate correctness of basic utilities
 **/

#include <gtest/gtest.h>
#include <utils.h>

/**
 * @brief Testing the initiallization process by making 4 different initiallizations
 * with different matrix sizes and check if the values inside are -1 or 1.
 * 
 **/
TEST(initiallization, init1) {
    bool valid = true;
    int n[4] = {100, 2000, 50000, 10000};
    int* arr;
    for(int i = 0; i < 4; i++)
    {
        arr = initiallize_model(n[i]);
        for(int j = 0; j < n[i]*n[i]; j++)
        {
            if(arr[j] != 1 && arr[j] != -1)
                valid=false;
        }   
        free(arr);
    }
    EXPECT_EQ(valid, true);
};

/**
 * @brief Testing the sign operator by testing the sign of different numbers both positive and negative
 * 
 **/
TEST(sign_test, s1) {
    int values[4] = {-100, -1, 1, 200};
    int ground_truth[4] = {-1, -1, 1, 1};
    bool valid = true;
    for(int i = 0; i < 4; i++)
    {
        if(sign(values[i]) != ground_truth[i])
            valid = false;
    }
    EXPECT_EQ(valid, true);
};

/**
 * @brief Death Testing of sign operator with 0 as input
 * 
 **/
TEST(sign_test, s2) {
    ASSERT_DEATH(sign(0), "");
};

/**
 * @brief testing of get_nsum with two inputs that grond truth have been calculated manually
 * 
 **/
TEST(get_nsum, s2) {
    int arr1[4] = {1, 1, -1, 1};    //neighboorhood's total sum = 
    int arr2[9] = {1, 1, -1, -1, 1, -1, 1, 1, -1};    //neighboorhood's total sum = 
    long sum1 = get_nsum(arr1, 2);
    long sum2 = get_nsum(arr2, 3);
    bool valid = true;
    if(sum1 != 10 || sum2 != 15)
        valid = false;

    EXPECT_EQ(valid, true);
};

/**
 * @brief testing of compare matrices
 * 
 **/
TEST(compare_matrice, succ_ver) {
    
    int mat1[9] = {1, 3, 4, 5, 1, -1, -100, 1, 1};
    int mat2[9] = {1, 3, 4, 5, 1, -1, -100, 1, 1};
    int valid = compare_matrices(mat1, mat2, 3);

    EXPECT_EQ(valid, 0);
};

/**
 * @brief testing of compare matrices, failing version
 * 
 **/
TEST(compare_matrice, fail_ver) {
    
    int mat1[9] = {1, 3, 4, 5, 1, -1, -100, 1, 1};
    int mat2[9] = {1, 3, 4, 5, 1, -1, 100, 1, 1};
    int valid = compare_matrices(mat1, mat2, 3);
    
    EXPECT_EQ(valid, 1);
};

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
