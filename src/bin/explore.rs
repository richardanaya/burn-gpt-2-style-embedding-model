extern crate simsimd;

fn main() {
    // Try different access patterns
    let vec1 = [1.0f32, 2.0, 3.0, 4.0];
    let vec2 = [2.0f32, 3.0, 4.0, 5.0];
    
    // Pattern 1: Direct module functions
    println!("Trying SimSIMD API...");
    
    // Try with explicit result type
    let result: Result<f32, _> = simsimd::cosine(&vec1, &vec2);
    match result {
        Ok(value) => println!("Cosine: {}", value),
        Err(e) => println!("Error: {:?}", e),
    }
}
}