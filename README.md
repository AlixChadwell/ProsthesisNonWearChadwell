# ProsthesisNonWearChadwell
This repository contains a non-wear algorithm designed to detect upper-limb prosthesis non-wear based on a wrist worn accelerometer.
It is based on the algorithm presented in Chadwell's PhD thesis. https://salford-repository.worktribe.com/output/1380938/the-reality-of-myoelectric-prostheses-how-do-emg-skill-unpredictability-of-prosthesis-response-and-delays-impact-on-user-functionality-and-everyday-prosthesis-use

The code was initially designed to run based on vector magitudes of x,y,z data from an Actigraph activity monitor so uses its version of an 'activity count' (https://s3.amazonaws.com/actigraphcorp.com/wp-content/uploads/2017/11/26205758/ActiGraph-White-Paper_What-is-a-Count_.pdf). It has been tested with other sensors such as Axivity, using Brond's conversion to Actigraph activity counts (https://journals.lww.com/acsm-msse/fulltext/2017/11000/generating_actigraph_counts_from_raw_acceleration.25.aspx - https://github.com/jbrond/ActigraphCounts).
