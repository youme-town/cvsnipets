私は日本語で会話しますが、プログラムのコメントは英語を使います。

私はDoxygenスタイルのコメントを使います。

私は@throwsより@exceptionを好みます。

私はコメントを書くとき、以下のテンプレートを参照しています。


以下、テンプレート

複数行コメント

```cpp
///
/// Comment
///
```

ファイルへのコメント

```cpp
/**
 * @file filename.h
 * @brief Brief description
 * @author Author
 * @date Date (start date?)
 */

```

関数へのコメント

```cpp
/**
 * @brief Brief description
 *
 * @details Detailed description
 * @param [out]    var1 Description of the parameter
 * @param [in]     var2 Description of the parameter
 * @param [in,out] var3 Description of the parameter
 * @par            Refer
 * - Global variable referenced: global_var1
 * - Global variable referenced: global_var2
 * @par            Modify
 * - Global variable modified: global_var3
 * - Global variable modified: global_var4
 * @return Description of the return value
 * @exception if not needed, none
 * @sa Functions to refer to can be linked here
 */
 int func(int var1, char *var2, char *var3[]){
		 return 0;
 }
```

変数へのコメント

```cpp
//! Comment for variable
int foo1 = 0;
```

グローバル変数へのコメント

```cpp
/** @var   global_var1
 *  @brief Description of the global variable
 */
int global_var1;

/** @var   global_var2
 *  @brief Description of the global variable
 *
 *  @details Detailed description of the global variable
 */
int global_var2;

```

マクロへのコメント

```cpp
/** @def
 *  Description of definition
 */
#define MAX_FOO 256

#define FOOFOO 758 /*!< Inline description of the definition */

/** @name Definition Group X
 *  Description of the grouped definitions
 */
/* @{ */
#define DDD 9AB /*!< Inline description of definition 1 */
#define EEE CDE /*!< Inline description of definition 2 */
/* @} */
```

列挙体へのコメント

```cpp
/** @enum eEnum1
 *  @brief Description of the enumeration
 */
typedef enum {
    aaa,

    //! Description of the enumeration value
    bbb,

    ccc, /*!< Example of adding a description for the enumeration value inline */
} eEnum1;

/** @enum eEnum2
 *  @brief Description of the enumeration
 *
 *  @details Detailed description of the enumeration
 */
typedef enum {
} eEnum2;
```

構造体へのコメント

```cpp
/** @struct tStruct1
 *  @brief Description of the structure
 */
typedef struct {
    //! Description of member1
    int member1;

    int member2; /*!< Inline description of member2 */
} tStruct1;

/** @struct tStruct2
 *  @brief Description of the structure
 *
 *  @details Detailed description of the structure
 */
typedef struct {
} tStruct2;

```

クラスへのコメント

```cpp
/** @class Class1
 *  @brief Description of the class
 */
class Class1 {
public:
    //! Default constructor
    Class1();

    /** Constructor
     *  @param [in] mem1 Description of mem1
     *  @param [in] mem2 Description of mem2
     */
    Class1(int mem1, int mem2); 

    /** Addition assignment operator
		 * 	@param [in] other `Class1` to add
		 *	@return Reference to the resulting object
		 */
		Class1& operator+=(const Class1& other);

    //! Description of member1
    int member1;

    int member2; /*!< Inline description of member2 */

    //! Description of method1
    int method1(int var1, int var2);

    /** Description of method2. 
     *
	 *	@details Detailed description
     *  @param[out]     var1    Description of var1
     *  @param[in]      var2    Description of var2
     *  @param[in,out]  var3    Description of var3
     *  @par            Refer
     *  - Referenced global variable global_var1
     *  - Referenced global variable global_var2
     *  @par            Modify
     *  - Modified global variable global_var3
     *  - Modified global variable global_var4
     *  @return         Success 0, Failure non-zero
     *  @exception      Exception. If none, specify none
     */
    int method2(int var1, int var2, int var3) {

    }
};

/** @class Class2
 *  @brief Description of the class
 *
 *  @details Detailed description of the class
 */
class Class2 {
};

```