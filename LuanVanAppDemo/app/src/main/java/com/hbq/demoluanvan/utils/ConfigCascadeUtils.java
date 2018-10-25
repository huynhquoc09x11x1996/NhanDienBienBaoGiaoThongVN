package com.hbq.demoluanvan.utils;

import android.app.Activity;
import android.content.Context;
import android.util.Log;

import com.hbq.demoluanvan.R;

import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class ConfigCascadeUtils {
    public static CascadeClassifier setup(Activity activity) {
        CascadeClassifier mCascade = null;
        try {
            InputStream inputStream = activity.getResources().openRawResource(R.raw.cascade2);
            File cascadeDir = activity.getDir("cascade2", Context.MODE_PRIVATE);
            File cascadeFile = new File(cascadeDir, "cascade2.xml");
            FileOutputStream os = new FileOutputStream(cascadeFile);
            byte[] buffer = new byte[4096];
            int byteread;
            while ((byteread = inputStream.read(buffer)) != -1) {
                os.write(buffer, 0, byteread);
            }
            inputStream.close();
            os.close();
            mCascade = new CascadeClassifier(cascadeFile.getAbsolutePath());
            if (mCascade.empty()) {
                Log.e("LuanVanTag", "load cascasde that bai!");
            }
            cascadeDir.delete();
            cascadeFile.delete();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return mCascade;
    }
}
