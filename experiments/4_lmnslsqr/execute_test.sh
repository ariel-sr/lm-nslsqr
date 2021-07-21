if [ $1 = "normal" ]
then
	# LM-nsLSQR with three layers (two layers of 3 bits and one layer of 2 bits)
	./exp4 80000 50000 3 3 3 2 0 > output-normal.txt
elif [ $1 = "uniform" ]
then
	# LM-nsLSQR with three layers (two layers of 3 bits and one layer of 2 bits)
	./exp4 80000 50000 3 3 3 2 1 > output-uniform.txt
elif [ $1 = "sparse1" ]
then
	# LM-nsLSQR with three layers (two layers of 3 bits and one layer of 2 bits)
	./exp4 80000 50000 3 3 3 2 2 > output-sparse1.txt
elif [ $1 = "sparse2" ]
then
	# LM-nsLSQR with three layers (two layers of 3 bits and one layer of 2 bits)
	./exp4 80000 50000 3 3 3 2 3 > output-sparse2.txt
elif [ $1 = "dense1" ]
then
	# LM-nsLSQR with three layers (two layers of 3 bits and one layer of 2 bits)
	./exp4 80000 50000 3 3 3 2 4 > output-dense1.txt
elif [ $1 = "dense2" ]
then
	# LM-nsLSQR with three layers (two layers of 3 bits and one layer of 2 bits)
	./exp4 80000 50000 3 3 3 2 5 > output-dense2.txt
else
	echo "Wrong problem selecte"
fi