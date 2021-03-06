var img = null;
var ws =null;
var num = '';


function onReceived(txt) {
  num = txt;
}
function clearImg() {
    
  background(102);
  img.background(255);
  num = '';
}
function sendImd(){
 img.loadPixels();
 pxs = new Array(img.pixels.length/4);
 for(let i=0;i<img.pixels.length;i+=4){
   pxs[i/4] = int((img.pixels[i] + img.pixels[i+1]+img.pixels[i+2])/3);
 }
 ws.send("imagem ppm");
 for(let i = 0;i<pxs.length;i++){
   ws.send(`${pxs[i]}`);
 }
 ws.send("end ppm");
 
 
  
}
var btClear = null;
var btCheck = null;
function setup() {
  createCanvas(1000, 500);
  img = createGraphics(28, 28);
  clearImg();
  btClear = new Button(500, 10, 80, 40, clearImg, 520, 30, 'clear');
  btCheck = new Button(590, 10, 80, 40, sendImd, 600, 30, 'Calcular');
  ws = new WebSocket("ws://127.0.0.1:13254/");
  ws.onmessage = function (event) {
    console.log('recebi ', event, event.data);
    num = event.data;
  };
  
    
  
  background(102);
}



function draw() {
  if (ws.readyState == ws.CLOSED){
    noLoop();
    background(102);
    fill(255);
    textSize(15);
    text("falha ao conectar com servidor",0,20);
    setTimeout(()=>{ window.location.reload();},2000);
    return;
  }
  if (mouseIsPressed === true) {
    var x = mouseX;
    var y = mouseY;
    var px = pmouseX;
    var py = pmouseY;
    if (y> 10 && y<490 && x> 10 && x<490) {
      x = map(x, 10, 490, 0, 28);
      y = map(y, 10, 490, 0, 28);
      py = map(py, 10, 490, 0, 28);
      px = map(px, 10, 490, 0, 28);
      img.line(x, y, px, py);
    }
    //line(mouseX, mouseY, pmouseX, pmouseY);
  }
  image(img, 10, 10, 480, 480);
  btClear.show();
  btCheck.show();
  push();
  textSize(250);
  fill(255);
  text(num, 650, 300);
  pop();
}

class Button {
  constructor(x, y, w, h, onPress, tx=null, ty=null, txt ='', fll=255, pressed = color(255, 0, 0)) {
    if (ty == null) {
      ty = y;
    }
    if (tx == null) {
      tx = x;
    }

    this.x = x;
    this.y = y;
    this.w = w;
    this.h = h;
    this.click = false;
    this.fll = fll;
    this.pss= pressed;
    this.func = onPress;
    this.txt = txt;
    this.tx = tx;
    this.ty = ty;
  }
  show() {
    push();
    if (mouseIsPressed) {
      if (mouseX > this.x && mouseX<this.x+this.w &&mouseY > this.y && mouseY<this.y+this.h) {
        fill(this.pss);
        this.click = true;
      }
    } else {
      fill(this.fll);
    }
    rect(this.x, this.y, this.w, this.h);
    fill(0);
    text(this.txt, this.tx, this.ty);
    pop();
    if (this.click && mouseIsPressed ==false) {

      if (this.func != null) {

        this.func();
      }
      this.click = false;
    }
  }
}
