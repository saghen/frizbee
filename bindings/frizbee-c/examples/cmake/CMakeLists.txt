cmake_minimum_required(VERSION 3.15)
project(frizbee-smoke C CXX)

find_package(frizbee CONFIG REQUIRED)

add_executable(frizbee-smoke-shared ../smoke.c)
target_link_libraries(frizbee-smoke-shared PRIVATE frizbee::frizbee)

add_executable(frizbee-smoke-static ../smoke.c)
target_link_libraries(frizbee-smoke-static PRIVATE frizbee::static)

add_executable(frizbee-smoke-cpp ../smoke.cpp)
target_compile_features(frizbee-smoke-cpp PRIVATE cxx_std_17)
target_link_libraries(frizbee-smoke-cpp PRIVATE frizbee::frizbee)
