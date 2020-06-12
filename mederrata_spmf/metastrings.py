import re

weight_code = """ lambda lambda_{0}_{1}, tau_{0}_{1}: tfd.Independent(
    tfd.Normal(
        loc=tf.zeros({3}, dtype={4}),
        scale=lambda_{0}_{1}*tau_{0}_{1}
    ),
    reinterpreted_batch_ndims={2}
)
"""

normal_code = """ tfd.Independent(
    tfd.Normal(
        loc={0},
        scale={1}
    ),
    reinterpreted_batch_ndims={2}
 )
"""


normal_lambda_code = """lambda {3}: tfd.Independent(
    tfd.Normal(
        loc={0},
        scale={1}
    ),
    reinterpreted_batch_ndims={2}
 )
"""

halfnormal_code = """ tfd.Independent(
    tfd.HalfNormal(
        scale={0}
    ),
    reinterpreted_batch_ndims={1}
 )
"""

halfnormal_lambda_code = """ lambda{2}: tfd.Independent(
    tfd.Normal(
        scale={0}
    ),
    reinterpreted_batch_ndims={1}
 )
"""

cauchy_code = """ tfd.Independent(
    tfd.HalfCauchy(
        loc=tf.zeros({0}, dtype={3}),
        scale={1}*tf.ones({0}, dtype={3})
    ),
    reinterpreted_batch_ndims={2}
)
"""

sq_igamma_code = """ lambda {1}: tfd.Independent(
    SqrtInverseGamma(
        concentration=0.5*tf.ones({0}, dtype={3}),
        scale=1.0/{1}
    ),
    reinterpreted_batch_ndims={2}
)
"""

igamma_code = """ tfd.Independent(
    tfd.InverseGamma(
        concentration=0.5*tf.ones({0}, dtype={3}),
        scale={1}*tf.ones({0}, dtype={3})
    ),
    reinterpreted_batch_ndims={2}
)
"""


def clean_str(s):
    return re.sub('[^0-9a-zA-Z]+', '_', str(s))