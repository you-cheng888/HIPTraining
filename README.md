# HIPTraining

GPU mat mul	"Shared Mem
(bytes)"	Block Dim & Grid Dim	"Kernel 
Running Time ns"
element-wise partitioning (row major)	0	(32,32,1), (32,32,1)	1247841
element-wise partitioning (col major)	0	(32,32,1), (32,32,1)	6625125
element-wise partitioning (col major, memory coalescing)	0	(32,32,1), (32,32,1)	1242242
tile-wise partitioning (row major)	8192	(32,32,1), (32,32,1)	518561
tile-wise partitioning (col major)	8192	(32,32,1), (32,32,1)	678562
tile-wise partitioning (col major, memory coalescing)	8192	(32,32,1), (32,32,1)	536641
![image](https://github.com/user-attachments/assets/77c26db9-5200-4ac4-b395-54c7428e0582)
