#include "Add.h"  
#include "gmock/gmock.h"

TEST(Add, 负数) {
	EXPECT_EQ(Add(-1, -2), -3);	//相等
	EXPECT_GT(Add(-4, -5), -6); //大于(会报错)
}

TEST(Add, 正数) {
	EXPECT_EQ(Add(1, 2), 3);	//相等
	EXPECT_GT(Add(4, 5), 6);	//大于
}



int main(int argc, char** argv)
{
	testing::InitGoogleTest(&argc, argv);//注册需要运行的所有测试用例
	return RUN_ALL_TESTS();	//执行测试,成功返回0失败返回1
}
