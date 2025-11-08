#pragma once

#include <string>
#include <vector>

std::vector<int> encode(std::string prompt);
std::string decode(std::vector<int> token_ids);
