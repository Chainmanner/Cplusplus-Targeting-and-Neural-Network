#include <iostream>

#ifndef ASSERT_H
#define ASSERT_H

#define assert_nonlethal( condition ) \
		if ( !(condition) )						\
			std::cout << "\nASSERTION FAILED\n - FILE " << __FILE__ << "\n - LINE " << __LINE__ << "\n - CONDITION \"" << #condition << "\"\n" << std::endl

#endif	// ASSERT_H