import numpy as np



def getCameraCalibration(name):

    c2_720p = {
        "camera_matrix": np.array(
                    [[5.84802266e+03, 0.00000000e+00, 7.07818118e+02],
                     [0.00000000e+00, 5.83163880e+03, 4.29697237e+02],
                     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
        "dist_coeffs": np.array([[-4.75882067e+00, -1.06155117e+01, -8.72666802e-02, -5.45345184e-02, .33269568e+03]])
    }

    legion_720p = {
        "camera_matrix": np.array([[1.04107436e+03, 0.00000000e+00, 6.37586954e+02],
                                   [0.00000000e+00, 1.04214507e+03, 3.81871266e+02],
                                   [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
        "dist_coeffs": np.array([[ 0.17656337, -0.72890008, -0.00347832, -0.000887, 0.83809996]])
    }

    ps3_eye_640p = {
        "camera_matrix": np.array([[529.90126483,            0, 308.40372814],
                                   [           0, 529.53982665, 253.66297328],
                                   [           0,            0,            1]]),
        "dist_coeffs": np.array([[-0.08668853,  0.07604643, -0.00035236,  0.00169945,  0.01110355]])
    }

    corrections = {
        "c2_720p": c2_720p,
        "legion_720p": legion_720p,
        "ps3_eye_640p": ps3_eye_640p
    }

    if(name in corrections):
        return corrections[name]
    else:
        print("calibration name not found, returning legion_720p")
        return corrections["legion_720p"]

