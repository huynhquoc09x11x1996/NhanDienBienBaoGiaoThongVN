package com.hbq.demoluanvan.activity;

import android.annotation.SuppressLint;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.Window;
import android.view.WindowManager;

import com.hbq.demoluanvan.R;
import com.hbq.demoluanvan.env.ImageUtils;
import com.hbq.demoluanvan.utils.Classifier;
import com.hbq.demoluanvan.utils.ConfigCascadeUtils;
import com.hbq.demoluanvan.utils.TensorFlowImageClassifier;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.util.HashMap;
import java.util.Timer;
import java.util.TimerTask;


public class Camera2Activity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    public static final String TAG = "HBQ_LV_CTU";

    private CameraBridgeViewBase mCameraView;

    private Mat mDst;
    private Mat mGray;
    private Mat mRgba;
    private MatOfRect mSigns;
    Rect mRect;
    private CascadeClassifier mCascade;
    private Classifier classifier;
    private static final int INPUT_SIZE = 299;
    private static final int IMAGE_MEAN = 128;
    private static final float IMAGE_STD = 128f;
    private static final String INPUT_NAME = "Mul";
    private static final String OUTPUT_NAME = "final_result";


    private static final String MODEL_FILE = "file:///android_asset/optimized_graph_neg.pb";
    private static final String LABEL_FILE =
            "file:///android_asset/retrained_labels_neg.txt";
    private HashMap<String, String> mClasses = new HashMap<>();


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);
        setContentView(R.layout.activity_camera2);
        initView();
        mClasses.put("001", " Cấm ngược chiều");
        mClasses.put("002", " Cấm dừng và đỗ xe");
        mClasses.put("003", " Trọng lượng cho phép");
        mClasses.put("004", " Giao với đường không ưu tiên");
        mClasses.put("005", " Chỗ ngoặc nguy hiễm");
        mClasses.put("006", " Nơi giao  vòng xuyến");
        mClasses.put("007", " Trẻ em/học sinh qua đường");
        mClasses.put("008", " Đường đi bộ");
        mClasses.put("009", " Giao nhau có tín hiệu đèn");
        mClasses.put("010", " Cấm rẽ");
        mClasses.put("011", " Nguy hiễm khác");
        mClasses.put("012", " Chỗ được phép quay xe");
        mClasses.put("013", " Giao nhau với đường ưu tiên");
        mClasses.put("014", " Đường không bằng phẳng");
        mClasses.put("015", " Người đi bộ cắt ngang");
        mClasses.put("016", " Chợ đông người");
        mClasses.put("017", " Cấm các phương tiện");
        mClasses.put("019", " Hướng phải vòng sang phải");
        mClasses.put("020", " Đi chậm lại");
        mClasses.put("023", " Cáp điện phía trên");
        mClasses.put("024", " Giao nhau với đường hẹp");
        mClasses.put("025", " Giữ cự ly tối thiểu giữa 2 xe");
        mClasses.put("unknown", "Unknown");
        classifier =
                TensorFlowImageClassifier.create(
                        getAssets(),
                        MODEL_FILE,
                        LABEL_FILE,
                        INPUT_SIZE,
                        IMAGE_MEAN,
                        IMAGE_STD,
                        INPUT_NAME,
                        OUTPUT_NAME);

        Timer timer = new Timer();
        timer.schedule(new TimerTask() {
            @Override
            public void run() {
                if (mRgba != null && mRgba.width() > 0 && mRgba.height() > 0 && mRect != null) {
                    Log.e("Today", "Area Rect: " + mRect.area());
                    Bitmap bm = Bitmap.createBitmap(mRgba.width(), mRgba.height(), Bitmap.Config.ARGB_8888);
                    Utils.matToBitmap(mRgba, bm);
                    bm = Bitmap.createBitmap(bm, mRect.x, mRect.y, mRect.width, mRect.height);
                    bm = Bitmap.createScaledBitmap(bm, 299, 299, true);
                    Imgproc.putText(mRgba, mClasses.get(classifier.recognizeImage(bm).get(0).getTitle()), new Point(mRect.x, mRect.y - 30), 3, 1, new Scalar(255, 0, 0, 255), 2);
                    Log.e("Today", "Label: " + mClasses.get(classifier.recognizeImage(bm).get(0).getTitle()));
                }
            }
        }, 0, 500);

    }


    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mGray = new Mat(height, width, CvType.CV_8UC1);
        mDst = new Mat(height, width, CvType.CV_8UC1);
        mSigns = new MatOfRect();
    }

    @Override
    public void onCameraViewStopped() {
        mDst.release();
        mRgba.release();
        mGray.release();
    }


    @SuppressLint("StaticFieldLeak")
    @Override
    public Mat onCameraFrame(final CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        if (mCascade != null) {
            mGray = inputFrame.gray();
            mRgba = inputFrame.rgba();
            mCascade.detectMultiScale(mGray, mSigns, 1.1, 3, 0, new Size(100, 100), new Size(800, 800));
            Rect[] signsArray = mSigns.toArray();
            Log.e("Today", "Number of sign: " + signsArray.length);
            if (signsArray.length > 0) {
                for (Rect rect : signsArray) {
                    if (rect.area() > 10000) {
                        mRect = ImageUtils.paddingMat(rect, 100);
                        Imgproc.rectangle(mRgba, new Point(mRect.x, mRect.y), new Point(mRect.x + mRect.width, mRect.y + mRect.height), new Scalar(255, 0, 0), 3);
                    }
                }
            }
            mGray.release();
            inputFrame.gray().release();
        }
        return mRgba;
    }


    @Override
    protected void onPause() {
        super.onPause();
        if (mCameraView != null) {
            mCameraView.disableView();
        }

    }

    @Override
    protected void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.e(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        } else {
            Log.e(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }


    private void initView() {
        System.loadLibrary("opencv_java3");
        mCameraView = findViewById(R.id.java_camera_view);
        mCameraView.setCvCameraViewListener(this);
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case SUCCESS: {
                    Log.e(TAG, "Opencv loaded succesfully!");
                    mCascade = ConfigCascadeUtils.setup(Camera2Activity.this);
                    mCameraView.enableView();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
            }
        }
    };
}