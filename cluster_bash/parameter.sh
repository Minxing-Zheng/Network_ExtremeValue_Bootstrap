#!/bin/bash


n_nodes="50 100 200 300 500 700 1000 1500 2000"
aug_type="none row_sum"
function="beta dir"


for a in $n_nodes;
  do
    for b in $aug_type;
      do
        for c in $function;
          do
            for d in {1..30};
              do
                  echo "$a $b $c $d"
              done;
          done;
      done;
  done;