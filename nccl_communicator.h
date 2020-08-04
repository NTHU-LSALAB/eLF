#pragma once

#include <memory>

#include "communicator.h"
#include "kvs.h"

namespace elf {

std::unique_ptr<Communicator>
create_nccl_communicator(KeyValueStore *kvs, const std::string &identifier, int rank, int size);

}
