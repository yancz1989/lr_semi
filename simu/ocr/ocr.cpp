#include "ocr.h"

#include <iostream>

#include <sys/types.h>
#include <dirent.h>
#include <sys/stat.h>
#include <stdarg.h>

#include <stdlib.h>
#include <sys/timeb.h>
#include <stdint.h>
#include <stdio.h>

using namespace std;

pthread_mutex_t  piReadSection;     // 互斥锁

int SearchFiles(const char * pszPath, string * pFileName)
{

  DIR *dir = opendir(pszPath);
  if (dir == NULL)
  {
    return 0;
  }

  int FileNum = 0;
  while (true)
  {
    struct dirent *s_dir = readdir(dir);
    if (s_dir == NULL) 
      break;

    if ((strcmp(s_dir->d_name, ".") == 0)||(strcmp(s_dir->d_name, "..") == 0))
      continue;

    char currfile[_MAX_PATH];
    sprintf(currfile, "%s/%s", pszPath, s_dir->d_name);
    struct stat file_stat;
    stat(currfile, &file_stat);
    if (!S_ISDIR(file_stat.st_mode))
    {
      int nLen = strlen(s_dir->d_name);
      if (nLen >= 3)
      {
        char * pszExt = strrchr(s_dir->d_name, '.');

        if (pszExt != NULL)
        {
          pFileName[FileNum++] = currfile;
        }
      }
    }
  }

  closedir(dir);
  return FileNum;
}

void UnicodeToUTF8(const unsigned char *src_unicode_str,
                   unsigned long src_unicode_str_len,
                   char *&dest_utf8_str,
                   unsigned long &dest_utf8_str_len)
{
    dest_utf8_str = new char[src_unicode_str_len * 3 + 1];
    memset(dest_utf8_str, 0, src_unicode_str_len * 3 + 1);

    unsigned short *InPutStr = (unsigned short *)src_unicode_str;
    int i = 0, offset = 0;
    for (i = 0; i < src_unicode_str_len; i++)
    {
        if (InPutStr[i] <= 0x0000007f)
        {
            dest_utf8_str[offset++] = (char)(InPutStr[i] & 0x0000007f);
        }
        else if (InPutStr[i] >= 0x00000080 && InPutStr[i] <= 0x000007ff)
        {
            dest_utf8_str[offset++] =
                (char)(0xC0 | ((InPutStr[i] >> 6) & 0x1F));
            dest_utf8_str[offset++] =
                (char)((InPutStr[i] & 0x0000003f) | 0x00000080);
        }
        else if (InPutStr[i] >= 0x00000800 && InPutStr[i] <= 0x0000ffff)
        {
            dest_utf8_str[offset++] =
                (char)(((InPutStr[i] & 0x0000f000) >> 12) | 0x000000e0);
            dest_utf8_str[offset++] =
                (char)(((InPutStr[i] & 0x00000fc0) >> 6) | 0x00000080);
            dest_utf8_str[offset++] =
                (char)((InPutStr[i] & 0x0000003f) | 0x00000080);
        }
    }
    dest_utf8_str[offset] = 0;
    dest_utf8_str_len = offset;
    return;
}


IREAD_ERR_CODE SetParam(IREAD_HANDLE hOCR, const OCR_PARAM & ocrParam)
{
  IREAD_ERR_CODE errorCode;

  //Set image type
  errorCode = iRead_SessionSetParam(hOCR, iREAD_PARAM_IMAGE_TYPE, (IREAD_PARAM_VALUE)ocrParam.iREAD_PARAM_IMAGE_TYPE);  
  if (IREAD_ERR_NONE != errorCode)
  {
    return errorCode;
  }

  //Set image binarize method
  errorCode = iRead_SessionSetParam(hOCR, iREAD_PARAM_IMAGE_BINARIZE_METHOD, (IREAD_PARAM_VALUE)ocrParam.iREAD_PARAM_IMAGE_BINARIZE_METHOD);  
  if (IREAD_ERR_NONE != errorCode)
  {
    return errorCode;
  }

  //Set layout method
  errorCode = iRead_SessionSetParam(hOCR, iREAD_PARAM_LAYOUT_METHOD, (IREAD_PARAM_VALUE)ocrParam.iREAD_PARAM_LAYOUT_METHOD);  
  if (IREAD_ERR_NONE != errorCode)
  {
    return errorCode;
  }

  //Set recog lang
  errorCode = iRead_SessionSetParam(hOCR, iREAD_PARAM_RECOG_LANG, (IREAD_PARAM_VALUE)ocrParam.iREAD_PARAM_RECOG_LANG);  
  if (IREAD_ERR_NONE != errorCode)
  {
    return errorCode;
  }

  //Set recog range
  errorCode = iRead_SessionSetParam(hOCR, iREAD_PARAM_RECOG_RANGE, (IREAD_PARAM_VALUE)ocrParam.iREAD_PARAM_RECOG_RANGE);  
  if (IREAD_ERR_NONE != errorCode)
  {
    return errorCode;
  }

  //Set recog range custom
  errorCode = iRead_SessionSetParam(hOCR, iREAD_PARAM_RECOG_CUSTOM_CHARS, (IREAD_PARAM_VALUE)ocrParam.iREAD_PARAM_RECOG_CUSTOM_CHARS);  
  if (IREAD_ERR_NONE != errorCode)
  {
    return errorCode;
  }

  //Set output full half
  errorCode = iRead_SessionSetParam(hOCR, iREAD_PARAM_OUTPUT_FULL_HALF, (IREAD_PARAM_VALUE)ocrParam.iREAD_PARAM_OUTPUT_FULL_HALF);  
  if (IREAD_ERR_NONE != errorCode)
  {
    return errorCode;
  }

  //Set output vert punc
  errorCode = iRead_SessionSetParam(hOCR, iREAD_PARAM_OUTPUT_VERT_PUNC, (IREAD_PARAM_VALUE)ocrParam.iREAD_PARAM_OUTPUT_VERT_PUNC);  
  if (IREAD_ERR_NONE != errorCode)
  {
    return errorCode;
  }

  //Set output disp code
  errorCode = iRead_SessionSetParam(hOCR, iREAD_PARAM_OUTPUT_DISP_CODE, (IREAD_PARAM_VALUE)ocrParam.iREAD_PARAM_OUTPUT_DISP_CODE);  
  if (IREAD_ERR_NONE != errorCode)
  {
    return errorCode;
  }

  //Set output result option
  errorCode = iRead_SessionSetParam(hOCR, iREAD_PARAM_OUTPUT_RESULT_OPTION, (IREAD_PARAM_VALUE)ocrParam.iREAD_PARAM_OUTPUT_RESULT_OPTION);
  return errorCode;
}

IREAD_ERR_CODE RecognizeRegions(const char * szFileName,
        const OCR_PARAM & ocrParam,
        const IREAD_HANDLE engine_handle,
        const IREAD_REGION * pRegions,
        IREAD_INT nRegionCount)
{
  IREAD_HANDLE hOCR;
  IREAD_IMAGE image;
  IREAD_RESULT pResult;

  IREAD_ERR_CODE errorCode = iRead_SessionStart(engine_handle, &hOCR);
  if (IREAD_ERR_NONE != errorCode)
  {
    return errorCode;
  }
  printf("iRead_SessionStart\n");
  
  errorCode = iRead_LoadImage(szFileName, &image);
  if (IREAD_ERR_NONE != errorCode)
  {
    iRead_SessionStop(hOCR);
    return errorCode;
  }
  printf("iRead_LoadImage\n");

        errorCode = SetParam(hOCR, ocrParam);
        if (IREAD_ERR_NONE != errorCode)
        {
                iRead_FreeImage(&image);
                iRead_SessionStop(hOCR);
                return errorCode;
        }
        printf("SetParam\n");

  errorCode = iRead_SetImage(hOCR, &image);
  if (IREAD_ERR_NONE != errorCode)
  {
    iRead_FreeImage(&image);
    iRead_SessionStop(hOCR);
    return errorCode;
  }
  printf("iRead_SetImage\n");

  errorCode = iRead_ResetRegions(hOCR, pRegions, nRegionCount); 
  if (IREAD_ERR_NONE != errorCode)
  {
    iRead_FreeImage(&image);
    iRead_SessionStop(hOCR);
    return errorCode;
  }
  printf("iRead_ResetRegions\n");

  errorCode = iRead_Recognize(hOCR, NULL, &pResult);
  if (IREAD_ERR_NONE != errorCode)
  {
    iRead_FreeImage(&image);
    iRead_SessionStop(hOCR);
    return errorCode;
  }
  printf("iRead_Recognize\n");

        char *text_utf8 = NULL;
        unsigned long text_utf8_len = 0;
        UnicodeToUTF8((const unsigned char *)pResult.pTextBuf,
                      pResult.nTextBufLen / 2 + 1,
                      text_utf8,
                      text_utf8_len);
  printf("len:%d, content:%s\n", text_utf8_len, text_utf8);
  delete[] text_utf8;

  IREAD_ERR_CODE errorCode1 = iRead_FreeImage(&image);
  errorCode1 = iRead_FreeResult(&pResult);
  errorCode1 = iRead_SessionStop(hOCR);
  return errorCode;
}

int main(int argc, char** argv)
{
  srand(time(NULL));
  //srand(737337);

  const char * pcszLibPath = "../Data/ResData";
  const char * pcszDataPath = "../Data/TestData";

  nFilesize = SearchFiles(pcszDataPath, strDataFileName);
  printf("********Test start!******\n");
  
  if (0 == nFilesize)
  {
    printf("No file in directory: %s .\nPlease add images to the directory.", pcszDataPath );
    getchar();
    return 0;
  }

  IREAD_HANDLE engine_handle = NULL;
  IREAD_ERR_CODE errorCode = iRead_Init(pcszLibPath, &engine_handle);
  if (errorCode != IREAD_ERR_NONE)
  {
    printf("Initial failed!\n");
    return 0;
  }

  IREAD_REGION regions[] = {{{0, 0, 1599, 1199}, IREAD_RGNTYPE_HORZTEXT, IREAD_LANGUAGE_CHINESE_CN, 0}};
  OCR_PARAM ocrParam = {IREAD_IMAGE_TYPE_NORMAL, IREAD_BINARIZE_ADAPTIVE, IREAD_LAYOUT_NEWSPAPER, IREAD_LANGUAGE_CHINESE_CN,
            IREAD_RECOG_RANGE_ALL, paramRecogCustomChars[0], IREAD_FH_FULL, IREAD_VP_NO_CHANGE, IREAD_DP_NO_CHANGE, IREAD_RESULT_TEXTBUF};
  RecognizeRegions("../Data/TestData/aa.jpg", ocrParam, engine_handle, regions, 1);

  errorCode = iRead_End(engine_handle);

  return 0;
}
