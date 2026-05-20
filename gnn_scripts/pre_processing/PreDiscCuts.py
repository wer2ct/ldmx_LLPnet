#The role of this script is to take a EaT signal or background file, and convert events that pass the Ecal and Hcal Energy requirements into serialized .npz files. This is a file format that is easier to work with for the purposes of creating graphs and image input.

#Use like --> python3 EaTSkimmer_eh.py <Input Root File> <Is Signal? (1 for true, 0 for false)> <file number> <ECal Threshold> <HCal Threshold> <type flag> <outfile location>

#Example python3 PreDiscCuts.py input.root 0 200 3160 4440 enriched_nuclear /home/wer2ct

#Imports
import awkward as ak
import numpy as np
import uproot
import sys

#Main Function
def main():
    #parse command line arguments
    input_file = sys.argv[1]
    is_signal = bool(int(sys.argv[2]))
    print(f"registered is signal {is_signal}")
    file_number = int(sys.argv[3])
    ecal_energy_threshold = int(sys.argv[4])
    hcal_energy_threshold = int(sys.argv[5])
    type_flag = sys.argv[6] #a tag that can indicate what type of background we have, or what batch of signal. 
    outfile = sys.argv[7]
    
    #Select proper pass names depending on signal or background
    if is_signal: 
        #for signal
        hcal_rec_pass = 'eat_vis'
        ecal_rec_pass = 'eat_vis'
        is_noise_name = 'is_noise_'
    else:
        #for background
        hcal_rec_pass = "eat"
        ecal_rec_pass = "eat"
        is_noise_name = "is_noise_"

    #Loop over events, evaluating cuts:
    print("Starting Event Processing")
    with uproot.open(input_file) as f:
        big_tree = f["LDMX_Events"]
        total_events = big_tree.num_entries

        #load branches into memory
        branches = {
            "ecal_energy": f"EcalRecHits_{ecal_rec_pass}.energy_",
            "ecal_noise": f"EcalRecHits_{ecal_rec_pass}.{is_noise_name}",
            "ecal_x": f"EcalRecHits_{ecal_rec_pass}.xpos_",
            "ecal_y": f"EcalRecHits_{ecal_rec_pass}.ypos_",
            "ecal_z": f"EcalRecHits_{ecal_rec_pass}.zpos_",
            "hcal_energy": f"HcalRecHits_{hcal_rec_pass}.energy_",
            "hcal_section": f"HcalRecHits_{hcal_rec_pass}.section_",
            "hcal_x": f"HcalRecHits_{hcal_rec_pass}.xpos_",
            "hcal_y": f"HcalRecHits_{hcal_rec_pass}.ypos_",
            "hcal_z": f"HcalRecHits_{hcal_rec_pass}.zpos_",
            "hcal_layer": f"HcalRecHits_{hcal_rec_pass}.layer_",
        }
        arrays = big_tree.arrays(branches.values(), library="ak")

        # Grab branches important to making cuts. 
        ecal_energy = arrays[branches["ecal_energy"]]
        ecal_noise = arrays[branches["ecal_noise"]]
        hcal_energy = arrays[branches["hcal_energy"]]
        hcal_section = arrays[branches["hcal_section"]]

        #ECal Energy
        ecal_effective_energy = ak.sum(ecal_energy * (~ecal_noise), axis=1) #ecal_energy * ~ecal_noise applies masking
        ecal_pass = ecal_effective_energy < ecal_energy_threshold

        #HCal Energy
        hcal_mask = hcal_section == 0
        hcal_effective = 12 * ak.sum(hcal_energy * hcal_mask, axis=1) #same deal, we create a mask of the section the multiply to apply it. 
        hcal_pass = hcal_effective > hcal_energy_threshold

        #Combined Cut
        event_mask = ecal_pass & hcal_pass #requires both Ecal and Hcal conditions met
        print(f"Total events: {len(event_mask)}")
        print(f"Passing events: {ak.sum(event_mask)}")
        print(f"Efficiency: {(ak.sum(event_mask) / len(event_mask))}")

        #Now apply the mask to all of our branches:
        hcal_x = arrays[branches["hcal_x"]][event_mask]
        hcal_y = arrays[branches["hcal_y"]][event_mask]
        hcal_z = arrays[branches["hcal_z"]][event_mask]
        hcal_layer = arrays[branches["hcal_layer"]][event_mask]
        hcal_energy_pass = 12*arrays[branches["hcal_energy"]][event_mask]
        hcal_section_pass = arrays[branches["hcal_section"]][event_mask]

        #Including the Ecal Hits
        ecal_x = arrays[branches["ecal_x"]][event_mask]
        ecal_y = arrays[branches["ecal_y"]][event_mask]
        ecal_z = arrays[branches["ecal_z"]][event_mask]
        ecal_energy_pass = arrays[branches["ecal_energy"]][event_mask]

        #Importantly want to preserve which event each hit belongs to!! 
        #We can broadcast the initial local index to the size of one our hit branches and then apply same mask. 
        placeholder_array = arrays[branches["hcal_x"]]
        event_ids = (ak.broadcast_arrays(ak.local_index(placeholder_array, axis=0), placeholder_array))[0]
        passed_ids = event_ids[event_mask]
        file_numbers = ak.broadcast_arrays(file_number, hcal_x)[0]
        signal_status = ak.broadcast_arrays(int(is_signal), hcal_x)[0]
        file_numbers_flat = ak.to_numpy(ak.flatten(file_numbers))
        event_ids_flat = ak.to_numpy(ak.flatten(passed_ids))
        signal_status_flat = ak.to_numpy(ak.flatten(signal_status))
        hcal_x_flat = ak.to_numpy(ak.flatten(hcal_x))
        hcal_y_flat = ak.to_numpy(ak.flatten(hcal_y))
        hcal_z_flat = ak.to_numpy(ak.flatten(hcal_z))
        hcal_energy_flat = ak.to_numpy(ak.flatten(hcal_energy_pass))
        hcal_layer_flat = ak.to_numpy(ak.flatten(hcal_layer))
        hcal_section_pass_flat = ak.to_numpy(ak.flatten(hcal_section_pass))

        #Do the same for the Ecal hits:
        ecal_placeholder_array = arrays[branches["ecal_x"]]
        ecal_event_ids = (ak.broadcast_arrays(ak.local_index(ecal_placeholder_array, axis=0), ecal_placeholder_array))[0]
        ecal_passed_ids = ecal_event_ids[event_mask]
        ecal_event_ids_flat = ak.to_numpy(ak.flatten(ecal_passed_ids))
        ecal_x_flat = ak.to_numpy(ak.flatten(ecal_x))
        ecal_y_flat = ak.to_numpy(ak.flatten(ecal_y))
        ecal_z_flat = ak.to_numpy(ak.flatten(ecal_z))
        ecal_energy_flat = ak.to_numpy(ak.flatten(ecal_energy_pass))

        #Combine into one big array
        # Array contents -> |Event Number*|Hcal x|Hcal y|Hcal z|Hcal layer|Hcal bar|Orientation|Hcal Energy|Signal Status*|File Number*| *indicates event-wise

        #A lot!
        hcal_output_array = np.column_stack((file_numbers_flat, 
                                             event_ids_flat, 
                                             signal_status_flat, 
                                             hcal_x_flat, 
                                             hcal_y_flat, 
                                             hcal_z_flat , 
                                             hcal_energy_flat, 
                                             hcal_layer_flat, 
                                             hcal_section_pass_flat))

        ecal_output_array = np.column_stack((ecal_event_ids_flat, 
                                             ecal_x_flat, 
                                             ecal_y_flat, 
                                             ecal_z_flat, 
                                             ecal_energy_flat))

        #make a little statistics array
        stats = np.array((len(event_mask), ak.sum(event_mask)))
        #Now save the array to a .npz file
        
        #make a string for if signal or not
        if int(sys.argv[2]) == 0:
            signal_string = 'background'
        if int(sys.argv[2]) == 1:
            signal_string = 'signal'
        
        np.savez(outfile+f'{type_flag}_filtered_{signal_string}_{file_number}.npz', 
                 hcal_hits_array = hcal_output_array, 
                 ecal_hits_array = ecal_output_array, 
                 stats_array = stats)

    print(f"Processed file, saved to {outfile}")

main()

            
            
        




        
    
    
    
