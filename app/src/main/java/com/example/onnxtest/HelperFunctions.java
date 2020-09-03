package com.example.onnxtest;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import java.util.HashSet;
import java.util.Iterator;
import java.util.Locale;
import java.util.Set;
import java.util.TreeSet;

public class HelperFunctions {
    public static class Box implements Comparable<Box>{ // int or float for coordinates
        public float confidence;
        public int category;
        public float x1, y1, x2, y2;
        public Box(float confidence, int category, float x1, float y1, float x2, float y2){
            this.confidence = confidence;
            this.category = category;
            this.x1 = x1;
            this.y1 = y1;
            this.x2 = x2;
            this.y2 = y2;
        }
        public Box(float confidence, int category, int x1, int y1, int x2, int y2){
            this.confidence = confidence;
            this.category = category;
            this.x1 = x1;
            this.y1 = y1;
            this.x2 = x2;
            this.y2 = y2;
        }

        @Override
        public int compareTo(Box box) {
            return this.confidence < box.confidence ? 1 : -1;
        }

        public static float intersect(Box box1, Box box2){
            float x_overlap = Math.min(box1.x2,box2.x2)-Math.max(box1.x1,box2.x1);
            float y_overlap = Math.min(box1.y2,box2.y2)-Math.max(box1.y1,box2.y1);
            return x_overlap*y_overlap;
        }

        public float intersect(Box box2){
            return intersect(this,box2);
        }

        public float area(){
            return (x2-x1)*(y2-y1);
        }

        public static float iou(Box box1, Box box2){
            float overlap = intersect(box1,box2);
            float union = box1.area()+box2.area()-overlap;
            return overlap/(union+1e-5f);
        }

        public float iou(Box box2){
            return iou(this,box2);
        }

        @NonNull
        @Override
        public String toString() {
            return String.format(Locale.getDefault(),
                    "[%f %d %f %f %f %f]",confidence,category, x1, y1, x2, y2);
        }
    }

    // Do NMS on set of boxes with iouThreshold default to 0.5f
    public static Set<Box> NMS(Set<Box> boxes, @Nullable Float iouThreshold){
        if(iouThreshold==null)iouThreshold = 0.5f;
        TreeSet<Box> sortedSet = new TreeSet<>(boxes);
        Set<Box> finalSet = new HashSet<>();
        while(!sortedSet.isEmpty()){
            // pop out first box with highest confidence
            Box highestConfidenceBox = sortedSet.pollFirst();
            finalSet.add(highestConfidenceBox);
            // loop through other boxes and remove if iou greater than threshold
            Iterator<Box> it = sortedSet.iterator();
            while(it.hasNext()){
                Box box = it.next();
                if(highestConfidenceBox.iou(box)>iouThreshold){
                    it.remove();
                }
            }
        }
        return finalSet;
    }
}
