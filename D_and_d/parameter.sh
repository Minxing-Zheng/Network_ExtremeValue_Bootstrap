#!/bin/bash

n_nodes="100 300 500"
#n_nodes="100 200"
aug_type="none avedegree"

num_run="50"

num_param="2 8"
#num_param="2 4"

function="beta dir"

#function="dir"

for a in $n_nodes;
  do
  for b in $aug_type;
    do
      for c in $num_run;
        do
          for d in $num_param;
            do
              for e in $function;
                do
                for f in {1..20};
                  do
                  echo "$a $b $c $d $e $f"
                done;
              done;
          done;
      done;
  done;
done;
