
#!/bin/bash
Path=./Submit/MaxDegree/Parameter
n_nodes="100 200 300 400 500 600 700 800 900 1000"
num_param="1 2 3 4 5"
function="beta dir"

for a in $n_nodes;do 
  for b in $num_param;do
  	for c in $function;do
  		echo "$a,$b,$c" >> "${Path}/node_${a}_param_${d}.txt"
  	 done;
  done;
done;