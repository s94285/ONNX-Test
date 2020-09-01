package com.example.onnxtest;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;

import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.google.android.material.snackbar.Snackbar;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.View;

import android.view.Menu;
import android.view.MenuItem;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.IOException;
import java.io.InputStream;
import java.util.Collections;
import java.util.Locale;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public class MainActivity extends AppCompatActivity {
    private TextView textView;
    private ImageView imageView;
    private static final String ONNX_MODEL_PATH = "file:///android_asset/version-RFB-320_simplified.onnx";
    private static final String TAG_INFO = "ONNX_TEST_INFO";
    // input dimension
    private static final int W = 320;
    private static final int H = 240;
    // Float model
    private static final float IMAGE_MEAN = 128f;
    private static final float IMAGE_STD = 128f;
    private static final float CONFIDENCE_THRESHOLD = 0.7f;
    // create input and output array
    private float[][][][] testData = new float[1][3][H][W];
    private float[][][] outputScores; // float[1][4420][2]
    private float[][][] outputBoxes;  // float[1][4420][4]
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        textView = findViewById(R.id.textView);
        textView.setText("Output");
        textView.setMovementMethod(new ScrollingMovementMethod());
        imageView = findViewById(R.id.imageView);

        runInference();

        FloatingActionButton fab = findViewById(R.id.fab);
        fab.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Toast.makeText(MainActivity.this,"test",Toast.LENGTH_SHORT).show();
                textView.setText("");
                runInference();
            }
        });
    }

    private void runInference() {
        try(OrtEnvironment env = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions opts = new OrtSession.SessionOptions()){
            opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT);
            opts.addNnapi();
//            opts.addCPU(false);

            Log.i(TAG_INFO,"Loading model from "+ONNX_MODEL_PATH);
            String actualFilename = ONNX_MODEL_PATH.split("file:///android_asset/")[1];
            InputStream modelStream = getAssets().open(actualFilename);
            byte[] modelBytes = new byte[modelStream.available()];
            modelStream.read(modelBytes);

            // read image file into bitmap
            Bitmap bmp = BitmapFactory.decodeStream(getAssets().open("13.jpg"));
            Bitmap resizedBmp = Bitmap.createScaledBitmap(bmp,W,H,false);
            imageView.setImageBitmap(resizedBmp);



            // create int array to store int pixel
            int[] pixelInt= new int[W*H];
            resizedBmp.getPixels(pixelInt,0,resizedBmp.getWidth(),0,0,resizedBmp.getWidth(),resizedBmp.getHeight());

            // fill in input array
            for(int i=0;i<H;i++){
                for(int j=0;j<W;j++){
                    int pixelValue = pixelInt[i*W+j];
                    testData[0][0][i][j] = (((pixelValue >> 16) & 0xFF)-IMAGE_MEAN)/IMAGE_STD; //R
                    testData[0][1][i][j] = (((pixelValue >> 8) & 0xFF)-IMAGE_MEAN)/IMAGE_STD; //G
                    testData[0][2][i][j] = ((pixelValue & 0xFF)-IMAGE_MEAN)/IMAGE_STD; //B
//                    testData[0][0][i][j] = 0f;
//                    testData[0][1][i][j] = 0f;
//                    testData[0][2][i][j] = 0f;
                }
            }

            try (OrtSession session = env.createSession(modelBytes,opts)){
                Log.i(TAG_INFO,"Inputs:");
                for (NodeInfo i : session.getInputInfo().values()) {
                    Log.i(TAG_INFO,i.toString());
                }
                Log.i(TAG_INFO,"Outputs:");
                for (NodeInfo i : session.getOutputInfo().values()) {
                    Log.i(TAG_INFO,i.toString());
                }
                // start inference
                String inputName = session.getInputNames().iterator().next();
                Log.i(TAG_INFO,"inputNodeName: "+inputName);
                try(OnnxTensor test = OnnxTensor.createTensor(env,testData);
                    OrtSession.Result output = session.run(Collections.singletonMap(inputName,test))){
                    Log.i(TAG_INFO,"Results: ");
                    for(int i=0;i<output.size();i++)
                        Log.i(TAG_INFO,"Output["+i+"]: "+output.get(i).toString());
                    outputScores = (float[][][]) output.get(0).getValue();
                    outputBoxes = (float[][][]) output.get(1).getValue();
//                    float[][][][] inputData = (float[][][][]) test.getValue();
//                    for(int i=0;i<10;i++){
//                        textView.append("\n");
//                        for(int j=0;j<10;j++){
//                            textView.append(inputData[0][2][j][i]+" ");
//                        }
//                    }
                    // Process outputs
                    processBoxes(outputScores,outputBoxes);
                }
            }


        } catch (OrtException | IOException e) {
            e.printStackTrace();
        }
    }

    private void processBoxes(float[][][] outputScores, float[][][] outputBoxes) {
        float[][] scores = outputScores[0];
        float[][] boxes = outputBoxes[0];

        for(int i=0;i<scores.length;i++){
//            if(scores[i][1]>0.2&&scores[i][1]<0.9999) {
                textView.append(String.format("\n %f %f", scores[i][0], scores[i][1]));
                textView.append(String.format(" %f %f %f %f", boxes[i][0] * W, boxes[i][1] * H, boxes[i][2] * W, boxes[i][3] * H));
//            }
            if(i>20)break;
        }
        Log.d(TAG_INFO,textView.getText().toString());
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }
}