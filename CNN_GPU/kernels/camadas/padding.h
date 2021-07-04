kV paddingfeed(Vector in,Vector out,
			   int txi,int tyi,
			   int txo,int tyo,
			   int t, int l ,
			   int k0){
	int k = get_global_id(0) + k0;
	int x, y, z;
	TensorRemap(k, x, y, z, txi, tyi)
	int s = TensorMap(x+t,y+l,z,txo,tyo);
	out[s] = in[k];
}
kV paddingBack(Vector gradNext,Vector gradin,
			   int txi,int tyi,
			   int txo,int tyo,
			   int t, int l , int k0){
	int k = get_global_id(0) + k0;
	int x, y, z;
	TensorRemap(k, x, y, z, txi, tyi)
	int s = TensorMap(x+t,y+l,z,txo,tyo);
	gradin[k] = gradNext[s];
}