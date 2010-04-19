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


exit 0
