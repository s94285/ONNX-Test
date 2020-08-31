package com.example.onnxtest;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;

import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.google.android.material.snackbar.Snackbar;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import android.util.Log;
import android.view.View;

import android.view.Menu;
import android.view.MenuItem;
import android.widget.TextView;
import android.widget.Toast;

import java.io.IOException;
import java.io.InputStream;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public class MainActivity extends AppCompatActivity {
    private TextView textView;
    private static final String ONNX_MODEL_PATH = "file:///android_asset/version-RFB-320_simplified.onnx";
    private static final String TAG_INFO = "ONNX_TEST_INFO";
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        textView = findViewById(R.id.textView);
        textView.setText("Output");

        try(OrtEnvironment env = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions opts = new OrtSession.SessionOptions()){
            opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT);
            opts.addNnapi();

            Log.i(TAG_INFO,"Loading model from "+ONNX_MODEL_PATH);
            String actualFilename = ONNX_MODEL_PATH.split("file:///android_asset/")[1];
            InputStream modelStream = getAssets().open(actualFilename);
            byte[] modelBytes = new byte[modelStream.available()];
            modelStream.read(modelBytes);

            try (OrtSession session = env.createSession(modelBytes,opts)){
                Log.i(TAG_INFO,"Inputs:");
                for (NodeInfo i : session.getInputInfo().values()) {
                    Log.i(TAG_INFO,i.toString());
                }
                Log.i(TAG_INFO,"Outputs:");
                for (NodeInfo i : session.getOutputInfo().values()) {
                    Log.i(TAG_INFO,i.toString());
                }

            }
        } catch (OrtException | IOException e) {
            e.printStackTrace();
        }

        FloatingActionButton fab = findViewById(R.id.fab);
        fab.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Toast.makeText(MainActivity.this,"test",Toast.LENGTH_SHORT).show();
            }
        });
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