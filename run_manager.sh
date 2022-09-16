runList=$1
cat $runList | while read _path _name 
do
# copy the heterogeneities model into the directory
cp $_path .
./submit.csh $_name
done
