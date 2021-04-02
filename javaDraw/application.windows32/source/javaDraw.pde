import java.util.*;
// Conexao socket
Client c = new Client("localhost", 5000);


// imagem para desenhar
PGraphics quadro;

// imagem para mostrar filtros
Tensor filtro;


// dimensao da imagem
int imx = 28, imy = 28;

// valores para posicionamento do quadro
int q_x0 =10, q_y0=10, q_w = 400, q_h = 400;
int back_color = 220;
int resposta = 0;
// Botoes
List<Button> botoes =  new ArrayList<Button>();
// variavel de controle para esperar requisito
boolean canCalcule = true;
void setup() {
  c.start();
  size(940, 540);
  quadro = createGraphics(imx, imy);
  filtro = new Tensor(this, q_x0+q_w+10, q_y0, q_w, q_h);
  background(back_color);
  quadro.beginDraw();
  quadro.background(255);
  quadro.endDraw();

  // LIMPAR QUADRO
  botoes.add(new Button(this, "CLEAR", q_x0, q_y0+q_h, 50, 40, new Runnable () {
    public void run() { 
      quadro.beginDraw();
      quadro.background(255);  
      quadro.endDraw();
      putAns();
      filtro.clear();
    }
  }
  ));

  // ENVIAR PARA REDE NEURAL
  botoes.add(new Button(this, "Calcule", q_x0+55, q_y0+q_h, 50, 40, new Runnable () {
    public void run() {     
      if(!canCalcule) return;
      canCalcule = false;
      quadro.loadPixels();
      c.sendBytes(Utils.ints12bytes(quadro.pixels));
    }
  }
  ));
  // MOSTRAR PROXIMO FILTRO
  botoes.add(new Button(this, "NEXT", q_x0+60+q_w, q_y0+q_h, 50, 40, new Runnable () {
    public void run() {     
       if(!canCalcule) return;
      if (filtro.dim +1 < filtro.z)filtro.dim++;
      else filtro.dim = 0;
      
      filtro.show(back_color);
    }
  }
  ));
  // MOSTRAR o FILTRO ANTERIOR
  botoes.add(new Button(this, "BACK", q_x0+10+q_w, q_y0+q_h, 50, 40, new Runnable () {
    public void run() {    
      if(!canCalcule) return;
      if (filtro.dim -1 >= 0)filtro.dim--;
      else filtro.dim = filtro.z - 1;
      filtro.show(back_color);
    }
  }
  ));
  c.setOnReceivedMsg(new Client.OnReceiveMsg() {
    public void receive(byte [] data) {
      //o primeiro byte é a resposta
      resposta = (int)(data[0]&0xff);
      putAns(resposta);
      // os proximos 3*4= 12 bytes sao as dimensoes x,y,z
      byte [] dimen  = new byte[12];
      System.arraycopy(data, 1, dimen, 0, 12);
      filtro.setDim( Utils.bytes2Ints(dimen));
      // os proximos bytes sao os dados do filtros (ja estao normalizados de 0 a 255)
      byte [] filter_data = new byte[data.length - 13];
      System.arraycopy(data, 13, filter_data, 0, data.length-13);
      filtro.setData(Utils.bytes2Ints1(filter_data));
      canCalcule = true;
      filtro.show(back_color);
    }
  }
  );
}

void draw() {
  image(quadro, q_x0, q_y0, q_w, q_h);
  drawImagem(quadro);
  for (Button b : botoes)
    b.check();
}
void drawImagem(PGraphics img) {
  if (mousePressed) {
    int x = (mouseX-q_x0)*(mouseX-q_w-q_x0);
    int y = (mouseY-q_y0)*(mouseY-q_h-q_y0);
    if (x<=0 && y<=0) {
      x =(int)map(mouseX, q_x0, q_w+q_x0, 0, imx);
      y = (int)map(mouseY, q_y0, q_h+q_y0, 0, imy);
      int xp =(int) map(pmouseX, q_x0, q_w+q_x0, 0, imx);
      int yp = (int)map(pmouseY, q_y0, q_h+q_y0, 0, imy);
      img.beginDraw();
      img.line(xp, yp, x, y);
      img.endDraw();
    }
  }
}
void putAns(int ans) {
  putAns();
  push();
  fill(0);
  stroke(0);
  textAlign(CENTER);
  textSize(50);
  text(""+ans, q_x0+q_w/2, q_y0+q_h+50+50);
  pop();
}
void putAns() {
  push();
  noStroke();
  fill(back_color);
  rect(q_x0, q_y0+q_h+50, q_w, 100);
  pop();
}
