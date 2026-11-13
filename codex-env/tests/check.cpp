#include <numeric>
#include <vector>
int main(){std::vector<int> v{1,2,3};return std::accumulate(v.begin(),v.end(),0)==6?0:1;}
