//
// Created by Henrique on 10/12/2021.
//

#ifndef GAB_IMAGE_WINDOW_H
#define GAB_IMAGE_WINDOW_H


#define GAB_GUI_SHOW_IMG 230194109765264
#define OnShowImg(hwnd,wparam,lparam)if(lparam == GAB_GUI_SHOW_IMG){DrawImage(hwnd,(Img)wparam);break;}
#define ShowImg()

#define TensorOnGUI(tensor,x,y,w,h)PostMessageA(GUI.hmain, WM_PAINT,(WPARAM)imgFromTensor(tensor,w,h,x,y), GAB_GUI_SHOW_IMG)
typedef struct Img_t {
	int8_t *data;
	int w, h;
	int x0, y0;
	int releasedata;

	void (*release)(struct Img_t **self);
} *Img;

void Img_release(Img *selfp) {
	if (!selfp)return;
	if (!*selfp)return;
	if ((*selfp)->releasedata && (*selfp)->data) {
		free((*selfp)->data);
	}
	free(*selfp);
	*selfp = NULL;
}

Img createImg(int x, int y, int w, int h, void *data, int releasedata) {
	Img self = calloc(1, sizeof(struct Img_t));
	self->data = data;
	self->x0 = x;
	self->y0 = y;
	self->w = w;
	self->h = h;
	self->releasedata = releasedata;
	self->release = Img_release;
	return self;
}

Img imgFromTensor(Tensor tensor,int largura,int altura,int x,int y) {
	ubyte *img = alloc_mem(largura, altura);
//	size_t largura = tensor->y * tensor->z;
//	size_t altura = tensor->x * tensor->w;
	int tw = largura/tensor->z;
	int th = altura/tensor->w;
	for (int w = 0; w < tensor->w; ++w) {
		for (int z = 0; z < tensor->z; ++z) {
			tensor->imagegray(tensor, img, largura, altura, tw, th, w *th, z * tw, z, w);
		}
	}
	return createImg(x,y,largura,altura,img,1);
}

void DrawImage(HWND hwnd, Img img) {
	PAINTSTRUCT ps;
	RECT r;
	GetClientRect(hwnd, &r);

	if (r.bottom == 0) {
		return;
	}
	HDC hdc = BeginPaint(hwnd, &ps);
	int px;
	for (int x = 0; x < img->h; ++x) {
		for (int y = 0; y < img->w; ++y) {
			px = img->data[x * img->w + y];
			SetPixel(hdc, y + img->x0, x + img->y0, RGB(px, px, px));
		}
	}
	pngGRAY("D:\\Henrique\\Rede-convolucional\\Gab\\bin\\teste.png",(void *)img->data,img->w,img->h);
	img->release(&img);
	printf("\t\tDrawImage\n");
	EndPaint(hwnd, &ps);
}

#endif //GAB_IMAGE_WINDOW_H
