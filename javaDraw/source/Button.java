import processing.core.*;
public class Button {
  private int x0, y0, w, h;
  private String name;
  private PApplet app;
  private Runnable run;
  private int cPress, cOn, cNorm, at;
  private int state = -1;
  private boolean clicked = false;
  private boolean change = true;
  public Button(PApplet  app, String name, int x0, int y0, int w, int h) {
    this.x0 = x0;
    this.y0 = y0;
    this.w = w;
    this.h = h;
    this.app = app;
    cPress = app.color(125, 255, 125);
    cNorm =  app.color(255);
    cOn = app.color(200);
    at= cNorm;
    this.name = name;
  }
  public Button(PApplet  app, String name, int x0, int y0, int w, int h, Runnable r) {
    this(app, name, x0, y0, w, h);
    run=r;
  }
  public void onClick(Runnable r) {
    run = r;
  }
  public void check() {
    at = cNorm;
    int nstate = 0;
    if (((app.mouseX-x0)*(app.mouseX-w-x0))<=0 && ((app.mouseY-y0)*(app.mouseY-h-y0))<=0) {
      at = cOn;
      nstate = 1;// em cima
      if (app.mousePressed) {
        clicked = true;
        at = cPress;
        nstate = 2;// pressionado
      } else if (clicked) {
        clicked = false ;
        at = cOn;
        nstate = 1;// em cima
        if (run != null) {
          run.run();
        }
      }
    } else if (clicked) {
      clicked = false ;
      nstate = 0;// fora
    }
    if (nstate != state) {
      app.push();
      app.fill(at);
      app.rect(x0, y0, w, h);
      app.fill(0);
      app.textAlign(app.CENTER);
      app.text(name, x0+w/2, y0+h/2);
      app.pop();
    }
    state = nstate;
  }
}
