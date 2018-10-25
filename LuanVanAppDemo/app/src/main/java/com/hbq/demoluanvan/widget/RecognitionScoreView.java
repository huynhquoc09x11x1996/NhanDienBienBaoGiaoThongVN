/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.hbq.demoluanvan.widget;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.drawable.BitmapDrawable;
import android.util.AttributeSet;
import android.util.TypedValue;
import android.view.View;

import com.hbq.demoluanvan.R;
import com.hbq.demoluanvan.utils.Classifier;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class RecognitionScoreView extends View implements ResultsView {
    private static final float TEXT_SIZE_DIP = 24;
    private List<Classifier.Recognition> results;
    private final float textSizePx;
    private final Paint fgPaint;
    private final Paint bgPaint;

    public RecognitionScoreView(final Context context, final AttributeSet set) {
        super(context, set);

        textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        fgPaint = new Paint();
        fgPaint.setTextSize(textSizePx);

        bgPaint = new Paint();

        bgPaint.setColor(0xcc4285f4);

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

        mResId.put("001", R.drawable.tsr001);
        mResId.put("002", R.drawable.tsr002);
        mResId.put("003", R.drawable.tsr003);
        mResId.put("004", R.drawable.tsr004);
        mResId.put("005", R.drawable.tsr005);
        mResId.put("006", R.drawable.tsr006);
        mResId.put("007", R.drawable.tsr007);
        mResId.put("008", R.drawable.tsr008);
        mResId.put("009", R.drawable.tsr009);
        mResId.put("010", R.drawable.tsr010);
        mResId.put("011", R.drawable.ic_action_info);
        mResId.put("012", R.drawable.tsr012);
        mResId.put("013", R.drawable.tsr013);
        mResId.put("014", R.drawable.tsr014);
        mResId.put("015", R.drawable.tsr015);
        mResId.put("016", R.drawable.tsr016);
        mResId.put("017", R.drawable.ic_action_info);
        mResId.put("019", R.drawable.tsr019);
        mResId.put("020", R.drawable.tsr020);
        mResId.put("023", R.drawable.tsr023);
        mResId.put("024", R.drawable.tsr024);
        mResId.put("025", R.drawable.tsr025);
        mResId.put("unknown", R.drawable.ic_action_info);

    }

    @Override
    public void setResults(final List<Classifier.Recognition> results) {
        this.results = results;
        postInvalidate();
    }

    private Map<String, String> mClasses = new HashMap<>();
    private Map<String, Integer> mResId = new HashMap<>();

    @SuppressLint({"DefaultLocale", "DrawAllocation"})
    @Override
    public void onDraw(final Canvas canvas) {
        final int x = 10;
        int y = (int) (fgPaint.getTextSize() * 1.5f);

        canvas.drawPaint(bgPaint);

        if (results != null) {
            Classifier.Recognition mMaxRecogition = results.get(0);
            canvas.drawText(mClasses.get(mMaxRecogition.getTitle()) + String.format(":%.2f", mMaxRecogition.getConfidence() * 100) + "%", x, y, fgPaint);
            Bitmap myLogo = ((BitmapDrawable) getResources().getDrawable(mResId.get(mMaxRecogition.getTitle()))).getBitmap();
            canvas.drawBitmap(myLogo, x, y + 50, null);
//            for (int i = 1; i < results.size(); i++) {
//                if (results.get(i).getConfidence() > results.get(i - 1).getConfidence()) {
//                    mMaxRecogition = results.get(i);
//                }
//            }
//            canvas.drawText(mClasses.get(mMaxRecogition.getTitle()) + String.format(":%.2f", mMaxRecogition.getConfidence() * 100) + "%", x, y, fgPaint);

        }
    }
}
