#!/bin/bash

for file in $(find output/ -name '*.pkl'  -maxdepth 1);
do
  if [ -f "$file" ]; then
    mv $file output/m_depth8.num_est100/"$file";
  fi
done
