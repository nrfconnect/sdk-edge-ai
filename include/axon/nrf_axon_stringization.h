/*
 * Copyright (c) 2024, Nordic Semiconductor ASA. All Rights Reserved.
 *
 * The information contained herein is confidential property of Nordic Semiconductor ASA.
 * The use, copying, transfer or disclosure of such information is prohibited except by
 * express written agreement with Nordic Semiconductor ASA.
 */


#pragma once

/*
 * macro jiu jitsu...
 *
 * Our goal is to generate the include file names from 3 tokens:
 * - file name root (ie,axon_mfcc_properties_)
 * - target name (ie, default)
 * - .h extension
 * The caller will provide the correct target name via the build system, and the proper
 * header files will be included.
 * Thanks stack-overflow (https://stackoverflow.com/questions/5256313/c-c-macro-string-concatenation)for below. 
*/

#define PPCAT2_NX(A, B) A ## B// concatenate w/o macro expanding
#define PPCAT2(A, B) PPCAT2_NX(A, B) // concatenate w/ macro expanding (macros are expanded by PPCAT before being passed to PPCAT_NX)
#define PPCAT3_NX(A, B, C) A ## B ## C // concatenate w/o macro expanding
#define PPCAT3(A, B, C) PPCAT3_NX(A, B, C) // concatenate w/ macro expanding (macros are expanded by PPCAT before being passed to PPCAT_NX)
#define STRINGIZE_NX(A) #A // convert to string litteral w/o macro expanding
#define STRINGIZE(A) STRINGIZE_NX(A) // convert to string litteral w/ macro expanding (macros are expanded by STRINGIZE before being passed to STRINGIZE_NX)
#define STRINGIZE_3_CONCAT(A,B,C) STRINGIZE(PPCAT3_NX(A,B,C)) // since the outer macro will expand A, B, C, call PPCAT_NX instead. 
#define STRINGIZE_2_CONCAT(A,B) STRINGIZE(PPCAT2_NX(A,B))

/*
 * concats a litteral with an expanded macro.
 * "litteral" will not be expanded, "macro" will be.
 * #define my_litteral FOO
 * #define my_macro BAR
 * STRINGIZE_CONCAT_LITTERAL_MACRO(my_litteral, my_macro) 
 * => my_litteralBAR
*/
#define MACRO_EXPAND(A) A
#define CONCAT_LITTERAL_MACRO(litteral, macro) PPCAT2_NX(litteral,MACRO_EXPAND(macro))
#define STRINGIZE_CONCAT_LITTERAL_MACRO(litteral, macro) STRINGIZE(litteral##MACRO_EXPAND(macro))
#define STRINGIZE_CONCAT_LITTERAL_MACRO_LITTERAL(litteral1, macro,litteral2) #litteral1##STRINGIZE(macro)##litteral2

/* examples 
#define T1 foo
#define T2 _bar
#define T3 .h

#include STRINGIZE_3_CONCAT(T1,T2,T3)
#include STRINGIZE_NX(PPCAT(T1, T2, T3)) // this becomes "PPCAT(T1, T2, T3)"
#include STRINGIZE(PPCAT_NX(foo, _bar, .h)) // this is "foo_bar.h" since neither foo nor .h are macros
#include STRINGIZE(PPCAT_NX(T1, T2, T3)) // this is "T1T2T3"
#include STRINGIZE(PPCAT(T1, T2, T3)) // this is "foo_bar.h"
*/
