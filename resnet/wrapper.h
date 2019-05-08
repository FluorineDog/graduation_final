#include "common.h"
class Dims : public std::vector<int> {
  public:
    using std::vector<int>::vector;
    operator int*(){
        return this->data();
    }
    operator int*() const{
        return *this;
    }
};
