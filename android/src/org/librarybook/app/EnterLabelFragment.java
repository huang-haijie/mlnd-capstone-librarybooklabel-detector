package org.librarybook.app;

import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.os.Bundle;
import android.app.Fragment;
import android.support.v4.app.DialogFragment;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.EditText;


/**
 * A simple {@link Fragment} subclass.
 * Activities that contain this fragment must implement the
 * {@link EnterLabelFragment.OnFragmentInteractionListener} interface
 * to handle interaction events.
 * Use the {@link EnterLabelFragment#newInstance} factory method to
 * create an instance of this fragment.
 */
public class EnterLabelFragment extends DialogFragment {

    private static final String TAG = "Enter Label Dialog";
    private String mLabelText;

    private OnFragmentInteractionListener mListener;

    public EnterLabelFragment() {
        // Required empty public constructor
    }


    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        Log.v(TAG, "On create.");

        AlertDialog.Builder builder = new AlertDialog.Builder(getActivity());
        LayoutInflater inflater = getActivity().getLayoutInflater();
        View view = inflater.inflate(R.layout.fragment_enter_label, null);
        final EditText editText = (EditText) view.findViewById(R.id.et_label_text);
        builder.setPositiveButton(android.R.string.ok, new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                Log.v(TAG, "OK clicked.");
                String text = editText.getText().toString();
                mLabelText = text;
                if (mListener != null)
                    mListener.onFragmentInteraction(text);
                else
                    Log.e(TAG, "No listener!");
            }
        });
        builder.setNegativeButton(android.R.string.cancel, null);
        if (mLabelText == null || mLabelText.isEmpty())
            editText.setHint(R.string.enter_label);
        else
            editText.setText(mLabelText);

        builder.setView(view);
        builder.setTitle(getString(R.string.enter_label));
        //builder.create();
        builder.show();
        editText.requestFocus();
    }


    /*@Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        // Inflate the layout for this fragment
        return inflater.inflate(R.layout.fragment_enter_label, container, false);
    }*/



    @Override
    public void onAttach(Context context) {
        super.onAttach(context);
        Log.v(TAG, "On Attach. " + context.toString());
        if (context instanceof OnFragmentInteractionListener) {
            mListener = (OnFragmentInteractionListener) context;
        } else {
            Log.e(TAG, context.toString() + " is not a listener!");
            throw new RuntimeException(context.toString()
                    + " must implement OnFragmentInteractionListener");
        }
    }


    /**
     * This interface must be implemented by activities that contain this
     * fragment to allow an interaction in this fragment to be communicated
     * to the activity and potentially other fragments contained in that
     * activity.
     * <p>
     * See the Android Training lesson <a href=
     * "http://developer.android.com/training/basics/fragments/communicating.html"
     * >Communicating with Other Fragments</a> for more information.
     */
    public interface OnFragmentInteractionListener {
        void onFragmentInteraction(String text);
    }
}
