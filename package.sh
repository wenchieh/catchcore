#!/usr/bin/env bash

PROJECT_VERSION=catchcore-1.0

rm ${PROJECT_VERSION}.tar.gz
rm -rf ${PROJECT_VERSION}
mkdir ${PROJECT_VERSION}
cp -R ./{demo.sh,package.sh,./example.tensor,./output,./src,Makefile,requirements,./run_*,Makefile,AUTHOR,README.*,LICENSE,user_guide.pdf} ./${PROJECT_VERSION}
tar cvzf ${PROJECT_VERSION}.tar.gz --exclude='._*' ${PROJECT_VERSION}
rm -rf ${PROJECT_VERSION}
echo done.