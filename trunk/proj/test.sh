#!/bin/bash
cd apps
cd app0
make
./test.sh
cd ../
cd ../
cd apps
cd app1
make
./test.sh
cd ../
cd ../
cd apps
cd app2
make
./test.sh
cd ../
cd ../
cd apps
cd app0doubling
make
./test.sh
cd ../
cd ../
cd apps
cd app2doubling
make
./test.sh
cd ../
cd ../
cd apps
cd app1doubling
make
./test.sh
cd ../
cd ../
cd apps
cd app0var
make
./test.sh



exit 0
