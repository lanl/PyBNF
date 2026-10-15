- **A `neg_bin` noise model at `location = mean` scores a negative prediction as the expected
  count 0**, as the median location already clamps its target. A prediction below `-r` (a
  falling cumulative output, say) was scored wrongly, and one of exactly `-r` divided by zero.
  No score of a prediction above `-r` changes. The adaptive sampler's output noise trajectory
  draws its negative-binomial noise from the same clamped mean, and writes it under NumPy 2,
  which refused the one-element array it assigned to each row.
