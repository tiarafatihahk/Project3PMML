# encryption_used
        user_choice_encryption = row.get(INPUT_COL_ENCRYPTION, "None")
        
        # PENTING: Petakan input 'None' dari file ke 'unencrypted' yang diharapkan model
        if user_choice_encryption == "None":
            user_choice_encryption = "unencrypted"
            
        for encryption_option in ENCRYPTION_OPTIONS:
            feature_name_encoded = f"encryption_used_{encryption_option}"
            final_model_inputs[feature_name_encoded] = 1 if user_choice_encryption == encryption_option else 0
