import java.io.*;
import java.nio.*;
public final class Utils {

  public static byte[] ints2Bytes(int[] values) {
    try {
      ByteArrayOutputStream baos = new ByteArrayOutputStream();
      DataOutputStream dos = new DataOutputStream(baos);
      for (int i=0; i < values.length; ++i) {
        dos.writeInt(values[i]);
      }
      return baos.toByteArray();
    }
    catch(IOException e) {
      e.printStackTrace();
      throw new Error("Nao foi possivel ler os bytes");
    }
  }   
  public static int[] bytes2Ints(byte[] values) {
    IntBuffer intBuf = ByteBuffer.wrap(values).order(ByteOrder.BIG_ENDIAN).asIntBuffer();
    int[] array = new int[intBuf.remaining()];
    intBuf.get(array);
    return array;
  }   
  public static int[] bytes2Ints1(byte[] values) {
    int[] array = new int[values.length];
    for (int i=0; i<values.length; i++) {
      array[i] = (int)(values[i] & 0xff);
    }
    return array;
  } 
  public static byte[] ints12bytes(int[] values) {
    byte[] array = new byte[values.length];
    for (int i=0; i<values.length; i++) {
      array[i] = (byte)(values[i] & 0xff);
    }
    return array;
  }   
  public static boolean waitUntil(long timeoutms, Condition condition) {
    long start = System.currentTimeMillis();
    while (condition.condition()) {
      if (System.currentTimeMillis() - start > timeoutms ) {
        return true;
      }
    }
    return false;
  }

  public static interface Condition {
    boolean condition();
  }
}
