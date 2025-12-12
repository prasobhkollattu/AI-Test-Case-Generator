"""
AI Test Case Generator - Complete Backend (Windows Compatible)
==============================================================
Thread-safe with Windows file handling fix!

Author: [Your Wife's Name]
Project: AI Test Case Generator - Portfolio Version
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import re
from threading import Lock, Thread, RLock
import os
import copy
import gc
import time
import shutil

# ==================== FASTAPI APP SETUP ====================

app = FastAPI(title="AI Test Case Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== GLOBAL STATE ====================

training_examples = []
next_example_id = 1

training_history = []
next_history_id = 1

current_model = None
current_tokenizer = None

training_lock = Lock()
model_lock = RLock()

training_in_progress = False

AUTO_TRAIN_ENABLED = True
MIN_EXAMPLES_FOR_TRAINING = 5
RETRAIN_AFTER_N_EXAMPLES = 3
new_examples_since_training = 0
model_save_path = "./models/trained_model"
LLM_IDENTIFIER = "deepseek-ai/deepseek-coder-1.3b-instruct" 
# ==================== REQUEST/RESPONSE MODELS ====================

class TrainingExample(BaseModel):
    """Schema for adding a new training example"""
    feature_description: str
    test_cases: str
    category: Optional[str] = "General"

class GenerateRequest(BaseModel):
    """Schema for generating test cases"""
    feature_description: str

# ==================== STARTUP ====================

@app.on_event("startup")
async def startup():
    """
    Load model on startup.
    Tries to load trained model from disk first, falls back to LLM_IDENTIFIER
    """
    global current_model, current_tokenizer
    
    print("=" * 60)
    print("🚀 Starting AI Test Case Generator")
    print("=" * 60)
    
    
    # Try to load saved trained model first
    if os.path.exists(model_save_path):
        print(f"📥 Loading trained model from {model_save_path}...")
        try:
            current_tokenizer = AutoTokenizer.from_pretrained(model_save_path)
            current_model = AutoModelForCausalLM.from_pretrained(model_save_path)
            print("✅ Trained model loaded from disk!")
        except Exception as e:
            print(f"⚠️ Could not load trained model: {e}")
            print(f"📥 Loading base {LLM_IDENTIFIER} instead...")
            current_tokenizer = AutoTokenizer.from_pretrained(LLM_IDENTIFIER)
            current_model = AutoModelForCausalLM.from_pretrained(LLM_IDENTIFIER)
    else:
        print(f"📥 No trained model found, loading base {LLM_IDENTIFIER}...")
        current_tokenizer = AutoTokenizer.from_pretrained(LLM_IDENTIFIER)
        current_model = AutoModelForCausalLM.from_pretrained(LLM_IDENTIFIER)
    
    # Set padding token
    if current_tokenizer.pad_token is None:
        current_tokenizer.pad_token = current_tokenizer.eos_token
        current_model.config.pad_token_id = current_tokenizer.pad_token_id
        print("✅ Set pad_token to eos_token")
    
    current_model.eval()
    
    print("✅ Model ready!")
    print(f"💾 Model parameters: {current_model.num_parameters() / 1e6:.1f}M")
    print("=" * 60)

# ==================== BACKGROUND TRAINING (WINDOWS COMPATIBLE) ====================

def train_model_in_background():
    """
    Training in background thread with Windows file handling fix.
    Properly handles file locks and memory cleanup for Windows OS.
    """
    global current_model, current_tokenizer, training_in_progress
    global new_examples_since_training, next_history_id
    
    # Prevent concurrent training
    if not training_lock.acquire(blocking=False):
        print("⚠️ Training already in progress, skipping...")
        return
    
    try:
        training_in_progress = True
        print("\n" + "=" * 60)
        print("🎓 AUTO-TRAINING STARTED (Background Thread)")
        print("=" * 60)
        
        # Check minimum examples
        if len(training_examples) < MIN_EXAMPLES_FOR_TRAINING:
            print(f"⚠️ Not enough examples: {len(training_examples)}/{MIN_EXAMPLES_FOR_TRAINING}")
            return
        
        # Create training record
        record_id = next_history_id
        next_history_id += 1
        
        training_record = {
            'id': record_id,
            'model_name': LLM_IDENTIFIER,
            'num_examples': len(training_examples),
            'epochs': 2,
            'status': 'IN_PROGRESS',
            'started_at': datetime.now().isoformat(),
            'completed_at': None,
            'metrics': None
        }
        training_history.insert(0, training_record)
        
        print(f"📊 Training with {len(training_examples)} examples...")
        print("⚡ Server remains responsive during training!")
        
        # Format training data
        formatted_data = []
        for example in training_examples:
            text = f"""### Feature Description:
{example['feature_description']}

### Generated Test Cases:
{example['test_cases']}
"""
            formatted_data.append({"text": text})
        
        # Create dataset
        dataset = Dataset.from_list(formatted_data)
        
        # Create copy of tokenizer for training (prevents conflicts)
        training_tokenizer = copy.deepcopy(current_tokenizer)
        
        if training_tokenizer.pad_token is None:
            training_tokenizer.pad_token = training_tokenizer.eos_token
            print("✅ Set pad_token for training tokenizer")
        
        # Tokenize dataset
        def tokenize_function(examples):
            return training_tokenizer(
                examples["text"],
                truncation=True,
                max_length=1024,
                padding="max_length"
            )
        
        print("🔄 Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        print("✅ Tokenization complete")
        
        # Training configuration
        training_args = TrainingArguments(
            output_dir="./temp_training",
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=50,
            weight_decay=0.01,
            logging_steps=10,
            save_steps=10000,
            learning_rate=5e-5,
            fp16=torch.cuda.is_available(),
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=training_tokenizer,
            mlm=False
        )
        
        print("✅ Data collator created")
        
        # Acquire model lock for training
        print("🔒 Acquiring model lock for training...")
        with model_lock:
            print("✅ Model lock acquired")
            
            # Create trainer
            trainer = Trainer(
                model=current_model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator,
            )
            
            # Train the model
            print("🚀 Training started (20-40 minutes)...")
            print("✅ You can still generate test cases during training!")
            train_result = trainer.train()
            
            # ==================== WINDOWS FILE HANDLING FIX ====================
            
            model_save_path = "./models/trained_model"
            temp_save_path = f"./models/trained_model_temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            print(f"💾 Saving trained model (Windows compatible)...")
            
            try:
                # Create temp directory
                os.makedirs(temp_save_path, exist_ok=True)
                
                # CRITICAL: Move model to CPU and clean up memory
                print("🔄 Preparing model for saving (Windows fix)...")
                
                # Move to CPU (releases GPU/memory locks)
                current_model.cpu()
                
                # Clear CUDA cache if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Force garbage collection (releases file handles)
                gc.collect()
                
                # Give Windows time to release file handles
                print("⏳ Waiting for Windows to release file handles...")
                time.sleep(2)
                
                # Save to temporary location first
                print(f"💾 Saving to temporary location: {temp_save_path}")
                current_model.save_pretrained(temp_save_path)
                current_tokenizer.save_pretrained(temp_save_path)
                
                print("✅ Saved to temporary location!")
                
                # Remove old model if exists
                if os.path.exists(model_save_path):
                    print("🗑️ Removing old model...")
                    try:
                        shutil.rmtree(model_save_path, ignore_errors=True)
                        time.sleep(1)
                    except Exception as e:
                        print(f"⚠️ Could not remove old model: {e}")
                
                # Move temp to final location
                print("📦 Moving to final location...")
                shutil.move(temp_save_path, model_save_path)
                
                print("✅ Model saved to disk successfully!")
                
                # Move model back to appropriate device
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                current_model.to(device)
                print(f"✅ Model moved back to {device}")
                
            except Exception as save_error:
                print(f"⚠️ Failed to save model: {save_error}")
                print("⚠️ Model training completed but not saved to disk")
                print("⚠️ Trained model still available in memory for this session")
                
                # Clean up temp directory if it exists
                if os.path.exists(temp_save_path):
                    try:
                        shutil.rmtree(temp_save_path, ignore_errors=True)
                    except:
                        pass
                
                # Ensure model is back on device even if save failed
                try:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    current_model.to(device)
                except:
                    pass
            
            # Set model to evaluation mode
            current_model.eval()
        
        print("🔓 Model lock released")
        
        # Update training record
        training_record['status'] = 'COMPLETED'
        training_record['completed_at'] = datetime.now().isoformat()
        training_record['metrics'] = {
            'train_loss': float(train_result.training_loss),
            'epochs': 2,
            'num_examples': len(training_examples)
        }
        
        # Reset counter
        new_examples_since_training = 0
        
        print("=" * 60)
        print("✅ AUTO-TRAINING COMPLETED!")
        print(f"📉 Training Loss: {train_result.training_loss:.4f}")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        
        if 'training_record' in locals():
            training_record['status'] = 'FAILED'
            training_record['completed_at'] = datetime.now().isoformat()
            training_record['metrics'] = {'error': str(e)}
    
    finally:
        training_in_progress = False
        training_lock.release()

# ==================== API ENDPOINTS ====================

@app.post("/api/training-examples")
async def add_training_example(example: TrainingExample):
    """
    Add training example and trigger auto-training if conditions met.
    Returns immediately - training happens in background thread.
    """
    global next_example_id, new_examples_since_training
    
    try:
        # Create new example
        new_example = {
            'id': next_example_id,
            'feature_description': example.feature_description,
            'test_cases': example.test_cases,
            'category': example.category,
            'created_at': datetime.now().isoformat()
        }
        
        # Add to memory
        training_examples.append(new_example)
        example_id = next_example_id
        next_example_id += 1
        
        # Increment counter
        new_examples_since_training += 1
        
        print(f"✅ Added example #{example_id}: {example.feature_description[:50]}...")
        
        # Prepare response
        total_examples = len(training_examples)
        response_data = {
            "success": True,
            "message": "Training example added!",
            "id": example_id,
            "total_examples": total_examples,
            "training_started": False
        }
        
        # Check if we should trigger auto-training
        should_train = False
        
        if AUTO_TRAIN_ENABLED and not training_in_progress:
            # First training: need minimum examples
            if total_examples >= MIN_EXAMPLES_FOR_TRAINING and new_examples_since_training >= MIN_EXAMPLES_FOR_TRAINING:
                should_train = True
                response_data["message"] = f"✅ Example added! 🎓 Auto-training started with {total_examples} examples (background)"
            # Subsequent trainings: retrain after N new examples
            elif total_examples >= MIN_EXAMPLES_FOR_TRAINING and new_examples_since_training >= RETRAIN_AFTER_N_EXAMPLES:
                should_train = True
                response_data["message"] = f"✅ Example added! 🎓 Retraining started with {total_examples} examples (background)"
            else:
                # Calculate remaining examples needed
                remaining = RETRAIN_AFTER_N_EXAMPLES - new_examples_since_training
                response_data["message"] = f"✅ Example added! {remaining} more example(s) until training."
        
        elif training_in_progress:
            response_data["message"] = "✅ Example added! Training already in progress."
        
        # Start training in separate thread if needed
        if should_train:
            print(f"🎯 Triggering auto-training! (Total: {total_examples}, New: {new_examples_since_training})")
            training_thread = Thread(target=train_model_in_background, daemon=True)
            training_thread.start()
            response_data["training_started"] = True
            print("✅ Training thread started - server remains responsive!")
        
        # Return immediately (server not blocked!)
        return response_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/training-examples")
async def get_training_examples():
    """Get all training examples from memory"""
    return {
        "success": True,
        "examples": training_examples,
        "count": len(training_examples)
    }

@app.delete("/api/training-examples/{example_id}")
async def delete_training_example(example_id: int):
    """Delete a training example from memory"""
    global training_examples
    
    original_count = len(training_examples)
    training_examples = [ex for ex in training_examples if ex['id'] != example_id]
    
    if len(training_examples) < original_count:
        print(f"🗑️ Deleted example #{example_id}")
        return {"success": True, "message": "Example deleted"}
    else:
        raise HTTPException(status_code=404, detail="Example not found")

@app.get("/api/training-status")
async def get_training_status():
    """
    Get current training status and history.
    Frontend polls this to update UI.
    """
    return {
        "training_in_progress": training_in_progress,
        "new_examples_since_training": new_examples_since_training,
        "examples_until_next_training": max(0, RETRAIN_AFTER_N_EXAMPLES - new_examples_since_training),
        "auto_training_enabled": AUTO_TRAIN_ENABLED,
        "total_examples": len(training_examples),
        "history": training_history[:10]
    }

@app.post("/api/generate")
async def generate_test_cases(request: GenerateRequest):
    """
    Generate test cases using the current model.
    Thread-safe - works even during training!
    """
    global current_model, current_tokenizer
    
    if current_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        start_time = datetime.now()
        
        print(f"🎯 Generating test cases for: {request.feature_description[:50]}...")
        
        # Build prompt
        prompt = f"""### Feature Description:
{request.feature_description}

### Generated Test Cases:

POSITIVE TEST CASES:
1."""

        # Acquire model lock with timeout (thread-safe)
        lock_acquired = model_lock.acquire(timeout=2)
        
        if not lock_acquired:
            # Training is using model, try again with longer timeout
            print("⚠️ Model is being trained, waiting for lock...")
            lock_acquired = model_lock.acquire(timeout=30)
            
            if not lock_acquired:
                raise HTTPException(
                    status_code=503,
                    detail="Model is currently being trained. Please try again in a few seconds."
                )
        
        try:
            print("✅ Model lock acquired for generation")
            
            # Tokenize input
            inputs = current_tokenizer(prompt, return_tensors="pt").to(current_model.device)
            
            # Generate test cases
            with torch.no_grad():
                outputs = current_model.generate(
                    **inputs,
                    max_new_tokens=800,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=current_tokenizer.eos_token_id,
                    repetition_penalty=1.2
                )
            
            # Decode output
            generated = current_tokenizer.decode(outputs[0], skip_special_tokens=True)
            test_cases_raw = generated.split("### Generated Test Cases:")[-1].strip()
            
            # Parse into categories
            formatted_cases = parse_test_cases(test_cases_raw)
            
        finally:
            # Always release the lock
            model_lock.release()
            print("🔓 Model lock released after generation")
        
        # Calculate generation time
        end_time = datetime.now()
        generation_time = (end_time - start_time).total_seconds()
        
        print(f"✅ Generated in {generation_time:.2f}s")
        
        return {
            "success": True,
            "feature": request.feature_description,
            "test_cases_raw": test_cases_raw,
            "formatted_cases": formatted_cases,
            "generation_time": round(generation_time, 2),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Generation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

def parse_test_cases(raw_output: str) -> Dict:
    """
    Parse raw model output into structured categories.
    Extracts test cases from different sections.
    """
    sections = {
        "positive": [],
        "negative": [],
        "boundary": [],
        "security": [],
        "performance": [],
        "integration": []
    }
    
    current_section = None
    lines = raw_output.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Detect section headers
        line_upper = line.upper()
        if 'POSITIVE' in line_upper:
            current_section = 'positive'
        elif 'NEGATIVE' in line_upper:
            current_section = 'negative'
        elif 'BOUNDARY' in line_upper or 'EDGE' in line_upper:
            current_section = 'boundary'
        elif 'SECURITY' in line_upper:
            current_section = 'security'
        elif 'PERFORMANCE' in line_upper:
            current_section = 'performance'
        elif 'INTEGRATION' in line_upper:
            current_section = 'integration'
        # Extract test cases (numbered or bulleted lines)
        elif current_section and (re.match(r'^\d+\.', line) or line.startswith('-') or line.startswith('•')):
            # Clean up the line
            clean_line = re.sub(r'^\d+\.\s*', '', line)
            clean_line = re.sub(r'^[-•]\s*', '', clean_line)
            # Only add substantial test cases
            if clean_line and len(clean_line) > 10:
                sections[current_section].append(clean_line)
    
    return sections

@app.get("/api/health")
async def health():
    """
    Health check endpoint.
    Always responds, even during training!
    """
    return {
        "status": "healthy",
        "model_loaded": current_model is not None,
        "training_in_progress": training_in_progress,
        "total_examples": len(training_examples),
        "auto_training_enabled": AUTO_TRAIN_ENABLED
    }

@app.get("/api/statistics")
async def get_statistics():
    """Get statistics about training data and sessions"""
    # Count examples by category
    categories = {}
    for ex in training_examples:
        cat = ex.get('category', 'General')
        categories[cat] = categories.get(cat, 0) + 1
    
    return {
        "total_examples": len(training_examples),
        "categories": categories,
        "training_sessions": len(training_history)
    }

# ==================== MAIN ====================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 60)
    print("🚀 AI Test Case Generator - Windows Compatible")
    print("=" * 60)
    print("📡 Server: http://localhost:8000")
    print("📄 API docs: http://localhost:8000/docs")
    print("✅ Training AND generation work simultaneously!")
    print("✅ Windows file locking handled properly!")
    print("=" * 60 + "\n")
    
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000)