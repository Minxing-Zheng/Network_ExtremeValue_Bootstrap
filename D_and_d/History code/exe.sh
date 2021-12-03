while IFS= read -r db
do
python /Users/app/Submit/MaxDegree/D_and_d/D_d.py $db
done < /Users/app/Submit/MaxDegree/D_and_d/parameter.txt
