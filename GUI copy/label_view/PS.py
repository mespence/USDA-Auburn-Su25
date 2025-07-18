import numpy as np
from scipy.ndimage import uniform_filter1d, median_filter1d

class ProbeSplitter:
    def classify_NP_vs_P_adaptive(self, recording, window=1000,
                                  min_probe_length=675, # Min length for a P region
                                  np_pad=415,           # Padding for NP regions around P
                                  std_dev_window=50,
                                  std_dev_threshold_NP=0.005, # Very low for NP (Z)
                                  derivative_window=10,
                                  derivative_threshold_NP=0.001, # Very low for NP (Z)
                                  voltage_epsilon_NP=0.001, # Small value above adaptive baseline for true NP (Z)
                                  adaptive_baseline_window=50000, # Long window for adaptive baseline (e.g., 5 seconds at 10kHz)
                                  ):
        """
        Classifies segments into 'NP' (Non-Probing, includes Z) and 'P' (Probing, includes G, C, D etc.)
        using an adaptive baseline.
        
        Outputs: A list of tuples (start_sample, end_sample, label_string)
                 where label_string is 'NP' or 'P'.
        """
        
        # Ensure recording is non-negative (pre-rectified assumption)
        abs_recording = np.abs(recording) 
        
        # --- Adaptive Baseline Estimation ---
        # A large median filter provides a robust estimate of the local minimum baseline.
        adaptive_baseline = median_filter1d(abs_recording, size=adaptive_baseline_window, mode='nearest')
        
        # 1. Smoothed voltage for general signal level
        smoothed = uniform_filter1d(abs_recording, size=window, mode='constant')

        # 2. Rolling Standard Deviation
        rolling_std = np.zeros_like(abs_recording, dtype=float)
        for i in range(len(abs_recording)):
            start = max(0, i - std_dev_window // 2)
            end = min(len(abs_recording), i + std_dev_window // 2 + 1)
            if end > start:
                rolling_std[i] = np.std(abs_recording[start:end])
            else:
                rolling_std[i] = 0

        # 3. Rolling Absolute Derivative
        abs_diff = np.abs(np.diff(abs_recording, prepend=abs_recording[0]))
        smoothed_abs_diff = uniform_filter1d(abs_diff, size=derivative_window, mode='constant')

        # --- Strict NP (Z) Identification using Adaptive Baseline and Dynamic Features ---
        # A region is NP if it's:
        # 1. Very low voltage (relative to its local adaptive baseline)
        # 2. AND Very low standard deviation (flatness)
        # 3. AND Very low derivative (minimal change, not abrupt)
        is_NP_strict = (smoothed < adaptive_baseline + voltage_epsilon_NP) & \
                       (rolling_std < std_dev_threshold_NP) & \
                       (smoothed_abs_diff < derivative_threshold_NP)
        
        # Everything that is NOT strictly NP, is considered P
        labels_raw = np.where(is_NP_strict, 'NP', 'P')

        # --- Segment Merging and Post-processing ---
        
        segments = []
        if len(labels_raw) == 0:
            return []

        # Initialize with the first segment
        current_start = 0
        current_label = labels_raw[0]

        for i in range(1, len(labels_raw)):
            if labels_raw[i] != current_label:
                segments.append((current_start, i, current_label))
                current_start = i
                current_label = labels_raw[i]
        
        # Add the last segment
        segments.append((current_start, len(labels_raw), current_label))

        # Second pass to apply min_probe_length and padding, and re-merge
        final_cleaned_segments = []
        if segments:
            current_start, current_end, current_label = segments[0]
            
            # Apply min_probe_length constraint for 'P' segments
            if current_label == 'P' and (current_end - current_start < min_probe_length):
                current_label = 'NP' # Reclassify short P as NP

            # Apply initial padding (only NP regions get padded to include surrounding context)
            padded_start = max(0, current_start - (np_pad if current_label == 'NP' else 0)) 
            padded_end = min(len(recording), current_end + (np_pad if current_label == 'NP' else 0))
            final_cleaned_segments.append((padded_start, padded_end, current_label))

            for i in range(1, len(segments)):
                next_start_raw, next_end_raw, next_label_raw = segments[i]
                
                # Apply min_probe_length constraint for 'P' segments
                if next_label_raw == 'P' and (next_end_raw - next_start_raw < min_probe_length):
                    next_label_raw = 'NP'

                # Apply padding (only NP regions get padded)
                next_padded_start = max(0, next_start_raw - (np_pad if next_label_raw == 'NP' else 0))
                next_padded_end = min(len(recording), next_end_raw + (np_pad if next_label_raw == 'NP' else 0))

                s_last, e_last, l_last = final_cleaned_segments[-1]

                # Merge logic:
                # If the next segment overlaps/touches the last, and:
                # 1. They have the same label: merge them.
                # 2. Different labels: prioritize 'P' over 'NP' if they overlap.
                if next_padded_start <= e_last: # Overlap or touch
                    if l_last == next_label_raw: # Same label, simple merge
                        final_cleaned_segments[-1] = (s_last, max(e_last, next_padded_end), l_last)
                    elif l_last == 'P': # Last was P, new is NP, keep as P (P dominates)
                        final_cleaned_segments[-1] = (s_last, max(e_last, next_padded_end), 'P')
                    elif next_label_raw == 'P': # Last was NP, new is P, change to P (P dominates)
                        final_cleaned_segments[-1] = (s_last, max(e_last, next_padded_end), 'P')
                    else: # Both are NP (should already be merged by l_last == next_label_raw) or other unexpected.
                          # This case should ideally not be hit if logic is perfect.
                        final_cleaned_segments[-1] = (s_last, max(e_last, next_padded_end), l_last)
                else: # No overlap, add as new segment
                    final_cleaned_segments.append((next_padded_start, next_padded_end, next_label_raw))
        
        # Final cleanup for probes (P regions) - extract only P segments
        probes_output = []
        for start, end, label in final_cleaned_segments:
            if label == 'P':
                probes_output.append((start, end))
        
        return probes_output