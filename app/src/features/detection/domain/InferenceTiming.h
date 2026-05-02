#pragma once

struct InferenceTiming {
    double preProcess  = 0.0;
    double inference   = 0.0;
    double postProcess = 0.0;
    double total       = 0.0;
};
