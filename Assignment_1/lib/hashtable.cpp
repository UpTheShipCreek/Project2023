#include "hashtable.h"

hFunction::hFunction(){
    this->V = Rand.generate_vector_normal(DIMENSIONS, MEAN, STANDARD_DEVIATION); // The N(0,1) distribution
    this->T = Rand.generate_double_uniform(0.0, this->W); 
    // So we were explicitly instructed to use the uniform(0,W) distribution for t and N(0,1) for the values of v, 
    // but to also ensure that (p*v + t) is not negative?
    // Why are we allowing negative values in the first place then? 
    // Why are we not shifting the normal distribution to N(1,1) and the uniform distribution to (1, W+1) 
    // so as to not have to worry about negative values?
}

double hFunction::evaluate_point(std::vector<double> p){ // h(p) = (p*v + t)/w
    double product = std::inner_product(p.begin(), p.end(), (this->V).begin(), 0); 
    
    double result = (product + this->T)/ this->W;

    return (int)std::floor(result); // Casting the result into into so that we may operate it with other ints
}




gFunction::gFunction(int k, int m){
    this->K = k;
    this->M = m;
    for(int i = 0; i < this->K; i++){
        std::shared_ptr<hFunction> h = std::make_shared<hFunction>(); // Create a new h function, I used a pointer might not need it, I just said to myself after OOP that I'd always use pointers instead of storing objects directly, something that I haven't followed here generally
        (this->H).push_back(h); // Save its pointer to the vector
        (this->R).push_back(Rand.generate_int_uniform(1, 100)); // Generate and save the r value
    }
}

int gFunction::evaluate_point(std::vector<double> p){
    int res;
    int sum = 0;
    for(int i = 0; i < this->K; i++){
        sum += (this->R)[i] * ((this->H)[i]->evaluate_point(p)); 
    }
    res = sum % M;

    if(res >= 0){
        //printf("%d gFunction::%s result = %d\n", __LINE__, __FUNCTION__, res);
        return res;
    }
    else{
        //printf("%d gFunction::%s result:%d + M:%d = %d\n", __LINE__, __FUNCTION__, res, M, res+M);
        return res+M; // Invoking the modular negation property in order to keep the result positive, this should be a very rare case after the changes in the p*v + t calculations 
    }
}



HashTable::HashTable(int num, int k, int m){ // Constructor
    this->NumberOfBuckets = num;
    this->HashFunction = std::make_shared<gFunction>(k, m);
}

bool HashTable::same_id(std::shared_ptr<ImageVector> image1, std::shared_ptr<ImageVector> image2){ // Compares the id of two images, used in the querying trick
    return NumberToId[image1->get_number()] == NumberToId[image2->get_number()];
}

void HashTable::insert(std::shared_ptr<ImageVector> image){ // Insert an image to the hash table and save its id
    std::vector<double> p = image->get_coordinates();
    int id = HashFunction->evaluate_point(p);

    NumberToId[image->get_number()] = id;

    int bucketId = id % NumberOfBuckets;

    if (Table.find(bucketId) == Table.end()) {
        Table[bucketId] = std::vector<std::shared_ptr<ImageVector>>();
    }

    Table[bucketId].push_back(image);
}

const std::vector<std::shared_ptr<ImageVector>>& HashTable::get_bucket_of_image(std::shared_ptr<ImageVector> image){ // Returns the bucket a specific image resides in 
    int bucketId = NumberToId[image->get_number()] % NumberOfBuckets;
    return Table[bucketId];
}
