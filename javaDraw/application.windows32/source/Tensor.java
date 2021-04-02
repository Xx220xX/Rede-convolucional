import processing.core.*;

public class Tensor {
  public int x=0, y=0, z=0;
  public int dim = 0;
  public int [] data;
  public int x0, y0, w, h;
  public PGraphics  quadroFiltros;
  private PApplet  app;
  public Tensor(PApplet  app, int x0, int y0, int w, int h) {
    this.app = app;
    this.x0 = x0;
    this.y0  = y0;
    this.h = h;
    this.w = w;
  }
  public void getPlaneXY(int z) {
    if (z>=this.z) throw new ArrayIndexOutOfBoundsException();
  }
  public void setDim(int [] dim) {
    if (dim.length != 3)throw new ArrayIndexOutOfBoundsException();
    x = dim[0];
    y = dim[1];
    z = dim[2];
  }
  public void setData(int [] src) {

    if (src.length != x*y*z)throw new ArrayIndexOutOfBoundsException();
    data = new int[src.length];
    System.arraycopy(src, 0, data, 0, data.length);
    dim = 0;
  }
  public void clear() {
    quadroFiltros = null;
    data = null;
    x=y=z = 0;
    dim= 0;
  }
  public void show(int back_color) {
    if(x*y*z == 0) return;
    if (quadroFiltros == null) {
      quadroFiltros = app.createGraphics(x, y);
    }
    quadroFiltros.beginDraw();
    quadroFiltros.background(255);
   
    for (int i=0; i<x; i++)
      for (int j=0; j<y; j++) {
        quadroFiltros.set(i, j, app.color(data[dim*(x*y)+j*y+i]));
      }
    quadroFiltros.endDraw();
    app.image(quadroFiltros, x0, y0, w, h);

    app.push();
    app.noStroke();
    app.fill(back_color);
    app.rect( x0+110, y0+h, 100, 40);
    app.pop();
    app.push();
    app.fill(0);
    app.stroke(0);
    app.textAlign(app.CENTER);
    app.textSize(25);
    app.text("["+dim+"] ("+x+"x"+y+")", x0+110+60, y0+h+20);
    app.pop();
    ;
  }
}
