import java.net.*;
import java.io.*;
import java.util.*;

public class Client extends Thread {
  private boolean canRun = true;
  private Socket cliente;
  private String host;
  private int port;
  InputStream input;
  OutputStream output ;
  private Client.OnReceiveMsg onReceive;
  public Client(String host, int port) {
    super();
    this.host = host;
    this.port = port;
  }
  private int available() {
    int ans = 0;
    try {
      ans = input.available();
    }
    catch(IOException e) {
      e.printStackTrace();
      throw new Error("Server has disconect");
    }
    return ans;
  }

  @Override
    public void run() {
    try {
      this.cliente = new Socket(this.host, this.port);
      input = cliente.getInputStream();
      output = cliente.getOutputStream();
    }
    catch(IOException e) {
      e.printStackTrace();
      throw new Error("Servidor não encontrado");
    }
    byte []b;
    while (canRun) {
      try {
        if (available()>4) {
          b = new byte[4];
          input.read(b);
          final int size = Utils.bytes2Ints(b)[0];

          Utils.waitUntil(3000, new Utils.Condition() { 
            public boolean condition() {
              return available() < size;
            }
          }
          );
          if (available() < size) {
            System.out.println("Error on receive byte, time out for wait "+size+" bytes");
            continue;
          }
          System.out.println("size "+size);
          b = new byte[size];
          input.read(b);
          OnReceivedMsg(b);
        }
        b = null;
      }
      catch(IOException e) {
        e.printStackTrace();
        canRun = false;
      }
    }
  }
  public void setOnReceivedMsg(OnReceiveMsg receive) {
    onReceive = receive;
  }
  private void OnReceivedMsg(byte [] data) {
    if ( onReceive != null) {
      onReceive.receive(data);
    } else {
      System.out.println("Foram ignorados "+data.length+" bytes recebidos");
    }
  }
  public void sendInts(int [] data) {
    byte [] bdata = Utils.ints2Bytes(data);
    byte [] blen = Utils.ints2Bytes(new int[]{bdata.length});
    byte [] bsend = new byte[bdata.length + blen.length];
    System.arraycopy(blen, 0, bsend, 0, blen.length);
    System.arraycopy(bdata, 0, bsend, blen.length, bdata.length);
    try {
      output.write(bsend);
    }
    catch(IOException e) {
      e.printStackTrace();
    }
  }
  public void sendBytes(byte [] bdata) {
    byte [] blen = Utils.ints2Bytes(new int[]{bdata.length});
    byte [] bsend = new byte[bdata.length + blen.length];
    System.arraycopy(blen, 0, bsend, 0, blen.length);
    System.arraycopy(bdata, 0, bsend, blen.length, bdata.length);
    try {
      output.write(bsend);
    }
    catch(IOException e) {
      e.printStackTrace();
    }
  }
  public static interface OnReceiveMsg {
    public void receive(byte [] data);
  }
}
