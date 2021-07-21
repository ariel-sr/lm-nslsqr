if [ $1 = "normal" ]
then
	# Four layers of 2 bits
	./exp3 80000 50000 4 2 2 2 2 2 1 0 > output1-normal.txt
	# Three layers: two layers of 3 bits and one layer of 2 bits
	./exp3 80000 50000 4 3 3 3 2 1 0 > output2-normal.txt
	# Two layers of 4 bits
	./exp3 80000 50000 2 4 4 1 0 > output3-normal.txt
	# One layer of 8 bits
	./exp3 80000 50000 1 8 1 0 > output4-normal.txt
	# LSQR
	./exp3 80000 50000 1 2 0 0 > output5-normal.txt
elif [ $1 = "uniform" ]
then
	# Four layers of 2 bits
	./exp3 80000 50000 4 2 2 2 2 2 1 1 > output1-uniform.txt
	# Three layers: two layers of 3 bits and one layer of 2 bits
	./exp3 80000 50000 4 3 3 3 2 1 1 > output2-uniform.txt
	# Two layers of 4 bits
	./exp3 80000 50000 2 4 4 1 1 > output3-uniform.txt
	# One layer of 8 bits
	./exp3 80000 50000 1 8 1 1 > output4-uniform.txt
	# LSQR
	./exp3 80000 50000 1 2 0 1 > output5-uniform.txt
elif [ $1 = "sparse1" ]
then
	# Four layers of 2 bits
	./exp3 80000 50000 4 2 2 2 2 2 1 2 > output1-sparse1.txt
	# Three layers: two layers of 3 bits and one layer of 2 bits
	./exp3 80000 50000 4 3 3 3 2 1 2 > output2-sparse1.txt
	# Two layers of 4 bits
	./exp3 80000 50000 2 4 4 1 2 > output3-sparse1.txt
	# One layer of 8 bits
	./exp3 80000 50000 1 8 1 2 > output4-sparse1.txt
	# LSQR
	./exp3 80000 50000 1 2 0 2 > output5-sparse1.txt
elif [ $1 = "sparse2" ]
then
	# Four layers of 2 bits
	./exp3 80000 50000 4 2 2 2 2 2 1 3 > output1-sparse2.txt
	# Three layers: two layers of 3 bits and one layer of 2 bits
	./exp3 80000 50000 4 3 3 3 2 1 3 > output2-sparse2.txt
	# Two layers of 4 bits
	./exp3 80000 50000 2 4 4 1 3 > output3-sparse2.txt
	# One layer of 8 bits
	./exp3 80000 50000 1 8 1 3 > output4-sparse2.txt
	# LSQR
	./exp3 80000 50000 1 2 0 3 > output5-sparse2.txt
elif [ $1 = "dense1" ]
then
	# Four layers of 2 bits
	./exp3 80000 50000 4 2 2 2 2 2 1 4 > output1-dense1.txt
	# Three layers: two layers of 3 bits and one layer of 2 bits
	./exp3 80000 50000 4 3 3 3 2 1 4 > output2-dense1.txt
	# Two layers of 4 bits
	./exp3 80000 50000 2 4 4 1 4 > output3-dense1.txt
	# One layer of 8 bits
	./exp3 80000 50000 1 8 1 4 > output4-dense1.txt
	# LSQR
	./exp3 80000 50000 1 2 0 4 > output5-dense1.txt
elif [ $1 = "dense2" ]
then
	# Four layers of 2 bits
	./exp3 80000 50000 4 2 2 2 2 2 1 5 > output1-dense2.txt
	# Three layers: two layers of 3 bits and one layer of 2 bits
	./exp3 80000 50000 4 3 3 3 2 1 5 > output2-dense2.txt
	# Two layers of 4 bits
	./exp3 80000 50000 2 4 4 1 5 > output3-dense2.txt
	# One layer of 8 bits
	./exp3 80000 50000 1 8 1 5 > output4-dense2.txt
	# LSQR
	./exp3 80000 50000 1 2 0 5 > output5-dense2.txt
else
	echo "Wrong problem selecte"
fi