#!/bin/bash

#Path=./Submit/MaxDegree/Parameter
#n_nodes="100 200 300 400 500 600 700 800 900 1000"
n_nodes="3000 5000"
#n_nodes="100 200"
function="beta dir"
ifA="True False"
num_param="2 3 4 5 6" #num dimension dir range(1,num_param+1)
#seed={1..100}
ifsub="True False"
subsize="10 30"

#800,000

#80,000
#if outfile does not exist

for a in $n_nodes;do #nodes_a_param_d.txt
  for b in $function;do
    for c in $ifA;do
      for d in $num_param;do #
        for e in {1..200};do #use $e results to calculate coverage rate N_trails
          for f in $ifsub;do 
            if [ $f != "True" ];then
              echo "$a,$b,$c,$d,$e,$f" 
            else
              for g in $subsize;do
                echo "$a,$b,$c,$d,$e,$f,$g"
              done
            fi

            #for g in $subsize;do
              #echo "$a,$b,$c,$d,$e,$f,$g">> "${Path}/node_${a}_param_${d}.txt"
              #echo "$a,$b,$c,$d,$e,$f,$g"
              #if [ $f == "True" ];then
                #echo "$a,$b,$c,$d,$e,$f,$g"
              #else
                #echo "$a,$b,$c,$d,$e,$f" 
              #fi
            done;
          done;
        done;
      done;
    done;
  done;
done;