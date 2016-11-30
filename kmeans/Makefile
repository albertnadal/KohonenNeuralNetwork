default: kmeans

clean:
	rm kmeans

kmeans: kmeans.c
	gcc -Wall -std=gnu9x -g -o kmeans kmeans.c -lm

test100: default
	./generate_test_data.sh 100 10000 | sort -t, -k 1,1n -k 2,2n > test100.txt
	./kmeans ./test100.txt 100 2 100
test1000: default
	./generate_test_data.sh 1000 10000 | sort -t, -k 1,1n -k 2,2n > test1000.txt
	./kmeans ./test1000.txt 1000 3 100
test10000: default
	./generate_test_data.sh 10000 10000 | sort -t, -k 1,1n -k 2,2n > test10000.txt
	./kmeans ./test10000.txt 10000 4 100

