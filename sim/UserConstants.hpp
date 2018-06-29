#pragma once

#include "../ext/ordered-map/tsl/ordered_map.h"

using std::string;


template<class K, class V>
class UserConstants
{
    tsl::ordered_map<K,V> m;
};