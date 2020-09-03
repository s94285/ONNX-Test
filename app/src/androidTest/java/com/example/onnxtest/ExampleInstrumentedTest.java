package com.example.onnxtest;

import android.content.Context;
import android.util.Log;

import androidx.test.platform.app.InstrumentationRegistry;
import androidx.test.ext.junit.runners.AndroidJUnit4;

import org.junit.Test;
import org.junit.runner.RunWith;

import java.util.HashSet;
import java.util.Set;
import java.util.TreeSet;

import static org.junit.Assert.*;

/**
 * Instrumented test, which will execute on an Android device.
 *
 * @see <a href="http://d.android.com/tools/testing">Testing documentation</a>
 */
@RunWith(AndroidJUnit4.class)
public class ExampleInstrumentedTest {
    @Test
    public void useAppContext() {
        // Context of the app under test.
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        assertEquals("com.example.onnxtest", appContext.getPackageName());
    }

    @Test
    public void boxBehaviors(){
        String TAG = "testBoxBehaviors";
        HelperFunctions.Box b1 = new HelperFunctions.Box(0.5f,0,-1,1,2,4);
        HelperFunctions.Box b2 = new HelperFunctions.Box(0.8f,0,0,0,3,3);
        assertEquals(4.0f,b1.intersect(b2),1e-6);
        assertEquals(4.0f/14,b1.iou(b2),1e-6);
        HashSet<HelperFunctions.Box> boxes = new HashSet<>();
        boxes.add(b1);boxes.add(b2);
        Log.d(TAG,"Before NMS: "+boxes);
        Set<HelperFunctions.Box> resultBoxes = HelperFunctions.NMS(boxes,0.2f);
        Log.d(TAG,"After NMS: "+resultBoxes);
    }
}